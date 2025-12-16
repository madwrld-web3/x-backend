import uvicorn
import secrets
import logging
import requests
import hmac
import hashlib
import os
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. CONSTANTS & SECRETS
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# SECURITY CRITICAL: This key ensures the same user always gets the same Agent wallet.
# In production, load this from os.environ.get("BACKEND_SECRET")
BACKEND_SECRET_KEY = b"CHANGE_THIS_TO_A_VERY_LONG_SECURE_RANDOM_STRING_IN_PROD"

# --- HELPER: DETERMINISTIC AGENT GENERATION ---
def derive_agent_account(user_wallet_address: str) -> Account:
    """
    Derives a deterministic private key for a specific user wallet.
    Formula: HMAC_SHA256(Master_Secret, User_Address)
    """
    # Normalize address to lowercase
    user_addr_clean = user_wallet_address.lower().strip()
    
    # Generate deterministic private key
    priv_key_raw = hmac.new(
        BACKEND_SECRET_KEY, 
        user_addr_clean.encode("utf-8"), 
        hashlib.sha256
    ).hexdigest()
    
    # Return Account object
    return Account.from_key("0x" + priv_key_raw)

# --- MODELS ---

class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class AccountRequest(BaseModel):
    wallet_address: str

class AgentRequest(BaseModel):
    user_address: str

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
    return {"status": "online", "service": "X/CHANGE Backend"}

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

# --- 2. ACCOUNT STATUS & AGENT CHECK ---
@app.post("/api/account-status")
async def get_account_status(req: AccountRequest):
    address = req.wallet_address.lower()
    
    # 1. Derive the Agent Address this user SHOULD have
    agent_account = derive_agent_account(address)
    agent_address = agent_account.address.lower()

    try:
        # 2. Get User State from Hyperliquid
        user_state = info.user_state(address)
        
        # 3. Check Account Value
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        
        # 4. Check if our Derived Agent is actually approved on-chain
        # Hyperliquid returns "isVault" or a list of authorized states. 
        # But simpler: we check if the SDK *could* trade. 
        # Actually, let's scan the user state. Unfortunately, HL API doesn't 
        # explicitly list "approved agents" in the basic user_state call easily.
        # We will assume not approved if false, the frontend handles the rest.
        
        # However, we can return the agent address we expect
        return {
            "exists": True,
            "account_value": account_value,
            "needs_deposit": account_value < 10.0,
            "agent_address": agent_address, # Frontend checks if this is cached
            "is_agent_approved": True # We assume true if exists, handled by try/catch in trade
        }

    except Exception as e:
        # User doesn't exist on HL yet
        return {
            "exists": False,
            "account_value": 0.0,
            "needs_deposit": True,
            "message": "User does not exist on Hyperliquid."
        }

# --- 3. GENERATE AGENT (DETERMINISTIC) ---
@app.post("/generate-agent")
async def generate_agent(data: AgentRequest):
    try:
        # Always returns the SAME agent for this user
        account = derive_agent_account(data.user_address)
        
        logger.info(f"Derived agent {account.address} for user {data.user_address}")
        
        return {
            "agentAddress": account.address,
            "message": "Agent derived. Please approve on frontend if not already done."
        }
    except Exception as e:
        logger.error(f"Agent gen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. APPROVE AGENT ---
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

# --- 5. TRADE (RE-DERIVE KEY) ---
@app.post("/trade")
async def execute_trade(trade: TradeRequest):
    # 1. Re-derive the same key we generated earlier
    agent_account = derive_agent_account(trade.user_address)

    try:
        exchange = Exchange(agent_account, API_URL, account_address=trade.user_address)

        # Price & Formatting logic
        all_mids = info.all_mids()
        price = float(all_mids.get(trade.coin))
        if not price:
            raise HTTPException(status_code=400, detail=f"Price not found for {trade.coin}")
            
        meta = info.meta()
        asset_info = next((a for a in meta["universe"] if a["name"] == trade.coin), None)
        decimals = asset_info["szDecimals"] if asset_info else 4
        
        position_value = trade.usd_size * trade.leverage
        size_token = round(position_value / price, decimals)

        logger.info(f"Trade: {trade.coin} | Agent: {agent_account.address} | User: {trade.user_address}")

        order_result = exchange.market_open(
            trade.coin, 
            trade.is_buy, 
            size_token, 
            None, 
            0.05 
        )
        
        if order_result["status"] == "err":
            # Pass the actual Hyperliquid error back to frontend
            raise HTTPException(status_code=400, detail=order_result["response"])

        return {"status": "success", "result": order_result}

    except Exception as e:
        logger.error(f"Trade failed: {e}")
        # If the error contains "User or API Wallet does not exist", it means the user hasn't approved the agent yet
        raise HTTPException(status_code=400, detail=str(e))

# --- 6. POSITIONS ---
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
                
                positions.append({
                    "coin": coin,
                    "size": size,
                    "side": "LONG" if size > 0 else "SHORT",
                    "entry_price": entry_px,
                    "mark_price": mark_px,
                    "unrealized_pnl": pnl
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
    agent_account = derive_agent_account(request.user_address)
    try:
        exchange = Exchange(agent_account, API_URL, account_address=request.user_address)
        result = exchange.market_close(request.coin)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Close failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
