# main.py
import uvicorn
import secrets
import logging
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from eth_account import Account

# Hyperliquid SDK
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="X/CHANGE", description="White Label Perpetual Exchange")

# 1. CORS (Kept your specific domains + added * for dev flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://x-change-frontend-theta.vercel.app",
        "https://x-change-frontend-theta.vercel.app/",
        "*" # Added for easier local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. IN-MEMORY AGENT STORAGE
# Stores { "user_wallet_address": "agent_private_key" }
# In production, replace this with a database (Redis/Postgres)
USER_AGENTS = {}

# 3. HYPERLIQUID CONNECTION
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)
BROKER_FEE_ADDRESS = "0x7505E9d72fc210958ca6A62CD1dcaacC6a41E0D4"

# --- DATA MODELS ---

class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class AgentRequest(BaseModel):
    user_address: str

class ApproveAgentRequest(BaseModel):
    user_wallet_address: str
    agent_address: str
    agent_name: str
    nonce: int
    signature: Dict[str, Any]

# Updated TradeRequest: No longer requires private key from frontend
class TradeRequest(BaseModel):
    user_address: str # Replaces user_main_wallet_address for consistency
    coin: str
    is_buy: bool
    sz: float        # Size in tokens (or USD, logic handled below)
    limit_px: float  # Limit price (set 0 for market)
    
class ClosePositionRequest(BaseModel):
    user_address: str
    coin: str

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "service": "X/CHANGE Backend", "agents_active": len(USER_AGENTS)}

# --- 1. MARKET DATA (Kept from your code) ---
@app.get("/markets", response_model=List[Asset])
async def get_markets():
    try:
        meta = info.meta()
        all_mids = info.all_mids()
        target_coins = {"BTC", "ETH", "SOL", "HYPE", "PAXG", "WIF", "PEPE"}
        assets = []
        
        for asset_info in meta["universe"]:
            symbol = asset_info["name"]
            # Filter for specific coins or HIP (Hyperliquid Index perps)
            if symbol in target_coins or symbol.startswith("HIP"):
                price = float(all_mids.get(symbol, 0))
                asset = Asset(
                    symbol=symbol,
                    name=symbol,
                    max_leverage=asset_info.get("maxLeverage", 50),
                    price=price
                )
                assets.append(asset)
        
        return sorted(assets, key=lambda x: x.symbol)
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch markets")

# --- 2. AGENT GENERATION (New Secure Flow) ---
@app.post("/generate-agent")
async def generate_agent(data: AgentRequest):
    """
    Generates a backend wallet for the user and stores the private key securely.
    Returns the address so the frontend can approve it.
    """
    try:
        priv_key = "0x" + secrets.token_hex(32)
        account = Account.from_key(priv_key)
        
        user_addr = data.user_address.lower()
        USER_AGENTS[user_addr] = priv_key
        
        logger.info(f"Generated agent {account.address} for user {user_addr}")
        
        return {
            "agentAddress": account.address,
            "message": "Agent generated. Please approve this address on frontend."
        }
    except Exception as e:
        logger.error(f"Agent gen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. APPROVE AGENT (Relay to Hyperliquid) ---
@app.post("/approve-agent")
async def approve_agent(request: ApproveAgentRequest):
    """
    The frontend signs the permission, backend relays it to Hyperliquid API.
    """
    try:
        logger.info(f"Relaying agent approval for {request.user_wallet_address}")
        
        payload = {
            "action": {
                "type": "approveAgent",
                "hyperliquidChain": "Mainnet",
                "signatureChainId": "0xa4b1",
                "agentAddress": request.agent_address,
                "agentName": request.agent_name,
                "nonce": request.nonce
            },
            "nonce": request.nonce,
            "signature": request.signature
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{API_URL}/exchange", json=payload, headers=headers)
        data = response.json()
        
        if response.status_code != 200 or "error" in data:
            logger.error(f"Hyperliquid Error: {data}")
            raise HTTPException(status_code=400, detail=f"Approval failed: {data.get('error', 'Unknown error')}")

        return {"status": "success", "data": data}

    except Exception as e:
        logger.error(f"Approval exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. TRADING (Secure: Uses stored Agent Key) ---
@app.post("/trade")
async def execute_trade(trade: TradeRequest):
    """
    Executes a trade using the securely stored Agent Wallet.
    """
    user_addr = trade.user_address.lower()
    
    # Check if we have an agent for this user
    if user_addr not in USER_AGENTS:
        raise HTTPException(status_code=400, detail="No agent found. Please click 'Authorize Agent' first.")
    
    # Retrieve the private key from memory
    agent_private_key = USER_AGENTS[user_addr]
    agent_account = Account.from_key(agent_private_key)

    try:
        # Initialize Exchange acting as the user
        exchange = Exchange(
            agent_account, 
            API_URL, 
            account_address=trade.user_address
        )

        logger.info(f"Executing trade for {user_addr} via Agent {agent_account.address}")

        # Execute Order (Limit or Market)
        # Note: If limit_px is 0, we can treat it as market, or use specific market method
        # Here we use the generic 'order' method which handles Gtc (Limit) well.
        
        # If you want Market Orders specifically:
        if trade.limit_px == 0:
             # Market Order Logic
             # We need to fetch price to calculate slippage if using usd_size, 
             # but here 'sz' is passed. Assuming 'sz' is in token units.
             print(f"Market buying {trade.sz} {trade.coin}")
             order_result = exchange.market_open(
                 trade.coin, 
                 trade.is_buy, 
                 trade.sz, 
                 None, 
                 0.01 # 1% Slippage
             )
        else:
            # Limit Order Logic
            order_result = exchange.order(
                name=trade.coin,
                is_buy=trade.is_buy,
                sz=trade.sz,
                limit_px=trade.limit_px,
                order_type={"limit": {"tif": "Gtc"}}
            )

        return {"status": "success", "result": order_result}

    except Exception as e:
        logger.error(f"Trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. POSITIONS & CLOSING (Kept from your code) ---
@app.get("/positions/{wallet_address}")
async def get_positions(wallet_address: str):
    try:
        user_state = info.user_state(wallet_address)
        positions = []
        
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        
        for pos in user_state.get("assetPositions", []):
            p = pos["position"]
            size = float(p["szi"])
            if size != 0:
                coin = p["coin"]
                entry_px = float(p["entryPx"])
                mark_px = float(info.all_mids().get(coin, entry_px))
                pnl = (mark_px - entry_px) * size if size > 0 else (entry_px - mark_px) * abs(size)
                value = abs(size) * mark_px
                
                positions.append({
                    "coin": coin,
                    "size": size,
                    "side": "LONG" if size > 0 else "SHORT",
                    "entry_price": entry_px,
                    "mark_price": mark_px,
                    "unrealized_pnl": pnl,
                    "pnl_percentage": (pnl / value * 100) if value else 0,
                    "leverage": p.get("leverage", {}).get("value", 1)
                })

        return {
            "positions": positions,
            "account_value": {
                "total_value": account_value,
                "withdrawable": float(margin_summary.get("withdrawable", 0))
            }
        }
    except Exception as e:
        logger.error(f"Position fetch error: {e}")
        return {"positions": [], "account_value": {}}

@app.post("/close-position")
async def close_position(request: ClosePositionRequest):
    user_addr = request.user_address.lower()
    
    if user_addr not in USER_AGENTS:
        raise HTTPException(status_code=400, detail="No agent found. Authorize first.")

    try:
        agent_private_key = USER_AGENTS[user_addr]
        account = Account.from_key(agent_private_key)
        exchange = Exchange(account, API_URL, account_address=request.user_address)
        
        logger.info(f"Closing position {request.coin} for {request.user_address}")
        result = exchange.market_close(request.coin)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Close failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
