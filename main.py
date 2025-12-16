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

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. IN-MEMORY AGENT STORAGE
USER_AGENTS = {}

# 3. HYPERLIQUID CONNECTION
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# --- MODELS ---

class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class AgentRequest(BaseModel):
    user_address: str

class AccountRequest(BaseModel):
    wallet_address: str

class ApproveAgentRequest(BaseModel):
    user_wallet_address: str
    agent_address: str
    agent_name: str
    nonce: int
    signature: Dict[str, Any]

class TradeRequest(BaseModel):
    user_address: str
    coin: str
    is_buy: bool
    usd_size: float
    leverage: int

class ClosePositionRequest(BaseModel):
    user_address: str
    coin: str

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "service": "X/CHANGE Backend", "agents_active": len(USER_AGENTS)}

# --- NEW: ACCOUNT STATUS CHECK (THE FIX) ---
@app.post("/api/account-status")
async def get_account_status(req: AccountRequest):
    """
    Checks if the user exists on Hyperliquid.
    If they have no history/balance, Hyperliquid returns an error.
    We catch this and return a 'exists': False flag to trigger the Deposit UI.
    """
    address = req.wallet_address
    try:
        # Try to fetch user state
        user_state = info.user_state(address)
        
        # If successful, calculate Account Value
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        
        return {
            "exists": True,
            "account_value": account_value,
            "needs_deposit": account_value < 10.0, 
            "raw_state": user_state
        }

    except Exception as e:
        # The SDK throws an error if the user has never interacted with Hyperliquid
        logger.info(f"New user detected (no HL history): {address}")
        return {
            "exists": False,
            "account_value": 0.0,
            "needs_deposit": True,
            "message": "User does not exist on Hyperliquid. Deposit required."
        }

# --- 1. MARKET DATA ---
@app.get("/markets", response_model=List[Asset])
async def get_markets():
    try:
        meta = info.meta()
        all_mids = info.all_mids()
        target_coins = {"BTC", "ETH", "SOL", "HYPE", "PAXG", "WIF", "PEPE"}
        assets = []
        
        for asset_info in meta["universe"]:
            symbol = asset_info["name"]
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

# --- 2. AGENT GENERATION (Secure) ---
@app.post("/generate-agent")
async def generate_agent(data: AgentRequest):
    try:
        priv_key = "0x" + secrets.token_hex(32)
        account = Account.from_key(priv_key)
        
        user_addr = data.user_address.lower()
        USER_AGENTS[user_addr] = priv_key
        
        logger.info(f"Generated agent {account.address} for user {user_addr}")
        
        return {
            "agentAddress": account.address,
            "message": "Agent generated. Please approve on frontend."
        }
    except Exception as e:
        logger.error(f"Agent gen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. APPROVE AGENT (Relay) ---
@app.post("/approve-agent")
async def approve_agent(request: ApproveAgentRequest):
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

# --- 4. TRADING ---
@app.post("/trade")
async def execute_trade(trade: TradeRequest):
    user_addr = trade.user_address.lower()
    
    if user_addr not in USER_AGENTS:
        raise HTTPException(status_code=400, detail="Agent not authorized. Click 'Activate Agent'.")
    
    agent_private_key = USER_AGENTS[user_addr]
    agent_account = Account.from_key(agent_private_key)

    try:
        exchange = Exchange(agent_account, API_URL, account_address=trade.user_address)

        all_mids = info.all_mids()
        price = float(all_mids.get(trade.coin))
        if not price:
            raise HTTPException(status_code=400, detail=f"Price not found for {trade.coin}")
            
        meta = info.meta()
        asset_info = next((a for a in meta["universe"] if a["name"] == trade.coin), None)
        decimals = asset_info["szDecimals"] if asset_info else 4
        
        position_value = trade.usd_size * trade.leverage
        size_token = round(position_value / price, decimals)

        logger.info(f"Executing Trade: {trade.coin} {size_token} units via Agent")

        order_result = exchange.market_open(
            trade.coin, 
            trade.is_buy, 
            size_token, 
            None, 
            0.05 
        )
        
        if order_result["status"] == "err":
             raise HTTPException(status_code=400, detail=order_result["response"])

        return {"status": "success", "result": order_result}

    except Exception as e:
        logger.error(f"Trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. POSITIONS ---
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
        raise HTTPException(status_code=400, detail="Agent not authorized.")

    try:
        agent_private_key = USER_AGENTS[user_addr]
        account = Account.from_key(agent_private_key)
        exchange = Exchange(account, API_URL, account_address=request.user_address)
        
        result = exchange.market_close(request.coin)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Close failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
