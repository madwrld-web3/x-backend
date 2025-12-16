from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from hyperliquid.info import Info
from hyperliquid.utils import constants

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hyperliquid Info (Mainnet)
# We do not need Exchange() here because the frontend trades directly
info = Info(constants.MAINNET_API_URL, skip_ws=True)

class AccountCheckRequest(BaseModel):
    wallet_address: str

@app.get("/")
def home():
    return {"status": "ok", "service": "x-change-backend"}

@app.get("/markets")
def get_markets():
    """
    Returns list of assets with current prices.
    """
    try:
        # Get metadata for all coins
        meta = info.meta()
        # Get current mid prices
        all_mids = info.all_mids()
        
        assets = []
        universe = meta["universe"]
        
        for coin_meta in universe:
            coin_name = coin_meta["name"]
            if coin_name in all_mids:
                price = float(all_mids[coin_name])
                assets.append({
                    "symbol": coin_name,
                    "price": price,
                    "szDecimals": coin_meta["szDecimals"]
                })
        
        return assets
    except Exception as e:
        print(f"Error fetching markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/account-status")
def check_account_status(req: AccountCheckRequest):
    """
    Checks if the wallet has ever deposited to Hyperliquid.
    """
    try:
        user_state = info.user_state(req.wallet_address)
        # If margin_summary is present, the account exists
        exists = user_state.get("marginSummary", None) is not None
        return {"exists": exists}
    except Exception as e:
        # If user doesn't exist, SDK often throws an error or returns empty
        print(f"Check status error (likely new user): {e}")
        return {"exists": False}

@app.get("/positions/{user_address}")
def get_positions(user_address: str):
    """
    Returns open positions and total account value.
    """
    try:
        user_state = info.user_state(user_address)
        
        raw_positions = user_state.get("assetPositions", [])
        margin_summary = user_state.get("marginSummary", {})
        
        positions = []
        for p in raw_positions:
            pos_data = p["position"]
            size = float(pos_data["szi"])
            if size == 0:
                continue
                
            entry_price = float(pos_data["entryPx"])
            coin = pos_data["coin"]
            unrealized_pnl = float(pos_data["unrealizedPnl"])
            
            positions.append({
                "coin": coin,
                "size": size,
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "side": "LONG" if size > 0 else "SHORT"
            })
            
        return {
            "positions": positions,
            "account_value": {
                "total_value": float(margin_summary.get("accountValue", 0)),
                "buying_power": float(margin_summary.get("withdrawable", 0)) 
            }
        }
    except Exception as e:
        print(f"Error fetching positions: {e}")
        # Return empty structure if failed (e.g. user not found)
        return {"positions": [], "account_value": {"total_value": 0, "buying_power": 0}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
