import uvicorn
import logging
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Hyperliquid SDK
from hyperliquid.info import Info
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

# HYPERLIQUID CONNECTION
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# CACHE ASSET INDICES (Required for signing trades)
# Hyperliquid orders use an integer "Asset ID" (e.g., ETH = 4), not the symbol.
meta = info.meta()
ASSET_MAP = {a["name"]: i for i, a in enumerate(meta["universe"])}
DECIMALS_MAP = {a["name"]: a["szDecimals"] for a in meta["universe"]}

# --- MODELS ---

class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class AccountRequest(BaseModel):
    wallet_address: str

class TradeRequest(BaseModel):
    user_address: str
    coin: str
    is_buy: bool
    usd_size: float
    leverage: int

class SubmitTradeRequest(BaseModel):
    action: Dict[str, Any]
    nonce: int
    signature: Dict[str, Any]

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
        # Refresh meta occasionally in prod, but static for now is fine
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

# --- 2. ACCOUNT STATUS ---
@app.post("/api/account-status")
async def get_account_status(req: AccountRequest):
    address = req.wallet_address.lower()
    try:
        user_state = info.user_state(address)
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        
        return {
            "exists": True,
            "account_value": account_value,
            "needs_deposit": account_value < 10.0
        }
    except Exception as e:
        return {
            "exists": False,
            "account_value": 0.0,
            "needs_deposit": True
        }

# --- 3. PREPARE TRADE (The Fix for No Agent) ---
@app.post("/prepare-trade")
async def prepare_trade(trade: TradeRequest):
    """
    Calculates the exact details for the trade so the Frontend can sign it.
    Returns the EIP-712 payload.
    """
    try:
        # 1. Get Price & Decimals
        all_mids = info.all_mids()
        price = float(all_mids.get(trade.coin))
        if not price:
            raise HTTPException(status_code=400, detail=f"Price not found for {trade.coin}")
        
        asset_index = ASSET_MAP.get(trade.coin)
        decimals = DECIMALS_MAP.get(trade.coin, 4)
        
        if asset_index is None:
            raise HTTPException(status_code=400, detail="Asset ID not found")

        # 2. Calculate Size in Tokens
        position_value = trade.usd_size * trade.leverage
        # Round to correct decimals for that specific coin
        size_token = round(position_value / price, decimals)
        
        if size_token == 0:
             raise HTTPException(status_code=400, detail="Trade size too small")

        # 3. Construct the "Action" Object (Hyperliquid Specific)
        # This matches the structure expected by the smart contract
        action = {
            "type": "order",
            "orders": [
                {
                    "a": asset_index,       # Asset Index
                    "b": trade.is_buy,      # Buy (True) / Sell (False)
                    "p": str(price * 1.05 if trade.is_buy else price * 0.95), # Limit price (5% slippage)
                    "s": str(size_token),   # Size
                    "r": False,             # Reduce Only
                    "t": {"limit": {"tif": "Gtc"}} # Order Type: Limit / Good Til Cancelled
                }
            ],
            "grouping": "na"
        }
        
        # 4. Return everything the frontend needs to sign
        return {
            "action": action,
            "asset_index": asset_index,
            "raw_price": price,
            "raw_size": size_token,
            "nonce": int(time.time() * 1000)
        }

    except Exception as e:
        logger.error(f"Prepare trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. SUBMIT TRADE (Relay Signed Transaction) ---
@app.post("/submit-trade")
async def submit_trade(req: SubmitTradeRequest):
    try:
        # Construct the final payload for the Exchange API
        payload = {
            "action": req.action,
            "nonce": req.nonce,
            "signature": req.signature
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{API_URL}/exchange", json=payload, headers=headers)
        data = response.json()
        
        if response.status_code != 200 or "error" in data:
            logger.error(f"Hyperliquid Trade Error: {data}")
            raise HTTPException(status_code=400, detail=data.get('error', 'Trade failed on-chain'))

        return {"status": "success", "result": data}

    except Exception as e:
        logger.error(f"Submit failed: {e}")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
