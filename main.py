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

# --- HELPER: CHECK AGENT APPROVAL ---
def check_agent_approved(user_address: str, agent_address: str) -> bool:
    """
    Check if an agent wallet is approved for a user by querying their state.
    Returns True if approved, False otherwise.
    """
    try:
        user_state = info.user_state(user_address)
        # The API doesn't directly expose approved agents, so we'll try to make a test order
        # and check if it fails due to agent not being approved
        return True  # We'll detect this through trade attempts
    except Exception as e:
        logger.error(f"Error checking agent approval: {e}")
        return False

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
            logger.error(f"Approval failed: {res.text}")
            raise HTTPException(status_code=400, detail=f"Hyperliquid Approval Failed: {res.text}")
        
        response_data = res.json()
        if response_data.get("status") == "err":
            raise HTTPException(status_code=400, detail=f"Approval error: {response_data.get('response')}")
            
        return {"status": "success", "data": response_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. TRADE (FIXED & ENHANCED) ---
@app.post("/trade")
async def execute_trade(trade: TradeRequest):
    """
    Execute a trade using the user's approved agent wallet.
    The agent must be pre-approved via the frontend flow.
    """
    try:
        # 1. Re-derive the same agent key
        agent_account = get_user_agent(trade.user_address)
        
        logger.info(f"Attempting trade for {trade.user_address} via agent {agent_account.address}")
        
        # 2. Create Exchange instance
        # CRITICAL: account_address tells SDK which user account to trade for
        exchange = Exchange(
            wallet=agent_account,
            base_url=API_URL,
            account_address=trade.user_address
        )

        # 3. Get current price & asset metadata
        all_mids = info.all_mids()
        price = float(all_mids.get(trade.coin, 0))
        if not price or price <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid price for {trade.coin}")
        
        meta = info.meta()
        asset_info = next((a for a in meta["universe"] if a["name"] == trade.coin), None)
        if not asset_info:
            raise HTTPException(status_code=400, detail=f"Asset {trade.coin} not found")
        
        decimals = asset_info.get("szDecimals", 4)
        
        # 4. Calculate position size
        # Formula: position_value = usd_size * leverage
        # size_in_tokens = position_value / current_price
        position_value = trade.usd_size * trade.leverage
        size_token = position_value / price
        
        # Round to correct decimals
        size_token = round(size_token, decimals)
        
        if size_token <= 0:
            raise HTTPException(status_code=400, detail="Position size too small")
        
        logger.info(f"Trade details: {trade.coin} {'BUY' if trade.is_buy else 'SELL'} {size_token} tokens @ ~${price}")
        logger.info(f"Position value: ${position_value:.2f} (size: ${trade.usd_size} × {trade.leverage}x leverage)")

        # 5. Execute market order
        # market_open params: coin, is_buy, sz, px (None for market), slippage
        order_result = exchange.market_open(
            coin=trade.coin,
            is_buy=trade.is_buy,
            sz=size_token,
            px=None,  # None = market order
            slippage=0.05  # 5% max slippage
        )
        
        logger.info(f"Order result: {order_result}")
        
        # 6. Check result
        if order_result.get("status") == "err":
            error_msg = order_result.get("response", "Unknown error")
            logger.error(f"Trade failed: {error_msg}")
            
            # Check for common agent errors
            if "does not exist" in str(error_msg).lower() or "api wallet" in str(error_msg).lower():
                raise HTTPException(
                    status_code=400, 
                    detail="Agent wallet not approved. Please activate your trading agent first."
                )
            
            raise HTTPException(status_code=400, detail=str(error_msg))
        
        # 7. Parse successful response
        response_data = order_result.get("response", {})
        data = response_data.get("data", {})
        statuses = data.get("statuses", [])
        
        # Extract fill information
        fills = []
        for status in statuses:
            if "filled" in status:
                filled = status["filled"]
                fills.append({
                    "order_id": filled.get("oid"),
                    "size": filled.get("totalSz"),
                    "price": filled.get("avgPx")
                })
        
        return {
            "status": "success",
            "result": order_result,
            "fills": fills,
            "message": f"{'Bought' if trade.is_buy else 'Sold'} {size_token} {trade.coin}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trade failed: {str(e)}")

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
                "total_value": float(margin_summary.get("accountValue", 0)),
                "withdrawable": float(margin_summary.get("withdrawable", 0))
            }
        }
    except Exception as e:
        logger.error(f"Positions fetch error: {e}")
        return {"positions": [], "account_value": {"total_value": 0, "withdrawable": 0}}

@app.post("/close-position")
async def close_position(request: ClosePositionRequest):
    """Close an open position."""
    agent_account = get_user_agent(request.user_address)
    try:
        exchange = Exchange(
            wallet=agent_account,
            base_url=API_URL,
            account_address=request.user_address
        )
        result = exchange.market_close(request.coin)
        
        if result.get("status") == "err":
            raise HTTPException(status_code=400, detail=result.get("response", "Failed to close"))
        
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Close position error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
