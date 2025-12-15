from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import requests
import time
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="X/CHANGE", description="White Label Perpetual Exchange")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://x-change-frontend-theta.vercel.app",
        "https://x-change-frontend-theta.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class ApproveAgentRequest(BaseModel):
    user_wallet_address: str
    agent_address: str
    agent_name: str
    nonce: int
    signature: Dict[str, Any]

class TradeRequest(BaseModel):
    user_agent_private_key: str
    user_main_wallet_address: str
    coin: str
    is_buy: bool
    usd_size: float
    leverage: int

class ClosePositionRequest(BaseModel):
    user_agent_private_key: str
    user_main_wallet_address: str
    coin: str

# --- HYPERLIQUID CONNECTION ---
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# YOUR FEE ADDRESS (Earnings go here)
BROKER_FEE_ADDRESS = "0x7505E9d72fc210958ca6A62CD1dcaacC6a41E0D4" 

@app.get("/")
async def root():
    return {"status": "online", "service": "X/CHANGE Backend"}

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

@app.post("/approve-agent")
async def approve_agent(request: ApproveAgentRequest):
    """
    Relays the User's signed 'Approve Agent' transaction to Hyperliquid.
    """
    try:
        logger.info(f"Relaying agent approval for {request.user_wallet_address}")
        
        payload = {
            "action": {
                "type": "approveAgent",
                "hyperliquidChain": "Mainnet",
                "signature": request.signature,
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

@app.post("/trade")
async def execute_trade(request: TradeRequest):
    """
    Executes a trade using the Agent Wallet.
    """
    try:
        account = Account.from_key(request.user_agent_private_key)
        exchange = Exchange(account, API_URL, account_address=request.user_main_wallet_address)

        # 1. Price & Precision
        all_mids = info.all_mids()
        price = float(all_mids.get(request.coin))
        if not price:
            raise HTTPException(status_code=400, detail=f"Price not found for {request.coin}")
            
        meta = info.meta()
        asset_info = next((a for a in meta["universe"] if a["name"] == request.coin), None)
        decimals = asset_info["szDecimals"] if asset_info else 4
        
        size_token = round(request.usd_size / price, decimals)

        # 2. Execute Trade (Market Order)
        logger.info(f"Executing Trade: {request.coin} {size_token} units")
        order_result = exchange.market_open(request.coin, request.is_buy, size_token, None, 0.05)
        
        if order_result["status"] == "err":
            raise HTTPException(status_code=400, detail=order_result["response"])

        # 3. OPTIONAL: Collect Fee (Currently commented out to prevent errors if balance is low)
        # To enable fees, uncomment the lines below. 
        # Note: This requires the user to have extra USDC to cover the transfer.
        
        # fee_amount = request.usd_size * 0.03  # 3% Fee
        # try:
        #     exchange.usd_transfer(BROKER_FEE_ADDRESS, fee_amount)
        #     logger.info(f"Fee collected: ${fee_amount}")
        # except Exception as fee_error:
        #     logger.error(f"Fee collection failed: {fee_error}")
        #     # We do NOT fail the trade if the fee fails, just log it.

        return {"status": "success", "result": order_result}

    except Exception as e:
        logger.error(f"Trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close-position")
async def close_position(request: ClosePositionRequest):
    try:
        account = Account.from_key(request.user_agent_private_key)
        exchange = Exchange(account, API_URL, account_address=request.user_main_wallet_address)
        
        result = exchange.market_close(request.coin)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Close failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
