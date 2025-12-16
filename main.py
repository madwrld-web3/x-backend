import uvicorn
import logging
import requests
import hmac
import hashlib
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. HYPERLIQUID CONNECTION
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# 2. MASTER SECRET (Crucial for Deterministic Agents)
# This ensures User X always gets Agent X.
# In production, keep this safe!
BACKEND_SECRET = b"WHITE_LABEL_EXCHANGE_MASTER_KEY_2024" 

# --- HELPER: DERIVE AGENT ---
def get_user_agent(user_address: str) -> Account:
    """
    Generates a stable, deterministic private key for a user.
    """
    user_clean = user_address.lower().strip()
    # HMAC-SHA256(Secret, UserAddress) -> Private Key
    priv_key_raw = hmac.new(BACKEND_SECRET, user_clean.encode(), hashlib.sha256).hexdigest()
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
                assets.append(Asset(
                    symbol=symbol,
                    name=symbol,
                    max_leverage=asset_info.get("maxLeverage", 50),
                    price=price
                ))
        return sorted(assets, key=lambda x: x.symbol)
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch markets")

# --- 2. ACCOUNT STATUS ---
@app.post("/api/account-status")
async def get_account_status(req: AccountRequest):
    address = req.wallet_address.lower()
    try:
        user_state = info.user_state(address)
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        return {"exists": True, "account_value": account_value, "needs_deposit": account_value < 10.0}
    except Exception:
        return {"exists": False, "account_value": 0.0, "needs_deposit": True}

# --- 3. GENERATE AGENT (Deterministic) ---
@app.post("/generate-agent")
async def generate_agent(data: AgentRequest):
    try:
        # Returns the SAME agent every time for this user
        account = get_user_agent(data.user_address)
        return {"agentAddress": account.address}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. APPROVE AGENT ---
@app.post("/approve-agent")
async def approve_agent(request: ApproveAgentRequest):
    try:
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
        res = requests.post(f"{API_URL}/exchange", json=payload, headers=headers)
        if res.status_code != 200:
            raise HTTPException(status_code=400, detail="Hyperliquid Approval Failed")
        return {"status": "success", "data": res.json()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. TRADE (RESTORED & FIXED) ---
@app.post("/trade")
async def execute_trade(trade: TradeRequest):
    # 1. Re-derive the same agent key
    agent_account = get_user_agent(trade.user_address)

    try:
        exchange = Exchange(agent_account, API_URL, account_address=trade.user_address)

        # 2. Get Price & Decimals
        all_mids = info.all_mids()
        price = float(all_mids.get(trade.coin))
        if not price:
            raise HTTPException(status_code=400, detail=f"Price not found for {trade.coin}")
            
        meta = info.meta()
        asset_info = next((a for a in meta["universe"] if a["name"] == trade.coin), None)
        decimals = asset_info["szDecimals"] if asset_info else 4
        
        # 3. Calculate Size
        position_value = trade.usd_size * trade.leverage
        size_token = round(position_value / price, decimals)

        logger.info(f"Trade: {trade.coin} {size_token} via {agent_account.address}")

        # 4. Execute
        order_result = exchange.market_open(
            trade.coin, trade.is_buy, size_token, None, 0.05
        )
        
        if order_result["status"] == "err":
            raise HTTPException(status_code=400, detail=order_result["response"])

        return {"status": "success", "result": order_result}

    except Exception as e:
        logger.error(f"Trade failed: {e}")
        # Pass exact error so frontend knows if it's an Agent issue
        raise HTTPException(status_code=400, detail=str(e))

# --- 6. POSITIONS ---
@app.get("/positions/{wallet_address}")
async def get_positions(wallet_address: str):
    try:
        user_state = info.user_state(wallet_address)
        positions = []
        margin_summary = user_state.get("marginSummary", {})
        
        for pos in user_state.get("assetPositions", []):
            p = pos["position"]
            size = float(p["szi"])
            if size != 0:
                coin = p["coin"]
                entry_px = float(p["entryPx"])
                mark_px = float(info.all_mids().get(coin, entry_px))
                pnl = (mark_px - entry_px) * size if size > 0 else (entry_px - mark_px) * abs(size)
                positions.append({
                    "coin": coin, "size": size, "side": "LONG" if size > 0 else "SHORT",
                    "entry_price": entry_px, "mark_price": mark_px, "unrealized_pnl": pnl
                })

        return {
            "positions": positions,
            "account_value": {
                "total_value": float(margin_summary.get("accountValue", 0)),
                "withdrawable": float(margin_summary.get("withdrawable", 0))
            }
        }
    except Exception as e:
        return {"positions": [], "account_value": {}}

@app.post("/close-position")
async def close_position(request: ClosePositionRequest):
    agent_account = get_user_agent(request.user_address)
    try:
        exchange = Exchange(agent_account, API_URL, account_address=request.user_address)
        result = exchange.market_close(request.coin)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
