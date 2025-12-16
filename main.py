import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from hyperliquid.info import Info
from hyperliquid.utils import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="X/CHANGE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hyperliquid connection - read-only info
API_URL = constants.MAINNET_API_URL
info = Info(API_URL, skip_ws=True)

# --- MODELS ---
class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class AccountRequest(BaseModel):
    wallet_address: str

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "service": "X/CHANGE"}

@app.get("/markets", response_model=List[Asset])
async def get_markets():
    """Get list of tradeable assets."""
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
        logger.error(f"Market error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch markets")

@app.post("/api/account-status")
async def get_account_status(req: AccountRequest):
    """Check if user has deposited and get account value."""
    address = req.wallet_address.lower()
    try:
        user_state = info.user_state(address)
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        
        logger.info(f"Account {address}: ${account_value}")
        
        return {
            "exists": account_value > 0,
            "account_value": account_value,
            "needs_deposit": account_value < 10.0
        }
    except Exception as e:
        logger.error(f"Account status error: {e}")
        return {"exists": False, "account_value": 0.0, "needs_deposit": True}

@app.get("/positions/{wallet_address}")
async def get_positions(wallet_address: str):
    """Get user's open positions and account value."""
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
        logger.error(f"Positions error: {e}")
        return {
            "positions": [],
            "account_value": {"total_value": 0, "withdrawable": 0}
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
