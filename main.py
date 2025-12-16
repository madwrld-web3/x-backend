from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hyperliquid.info import Info
from hyperliquid.utils import constants

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use Mainnet Info
info = Info(constants.MAINNET_API_URL, skip_ws=True)

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/markets")
def get_markets():
    try:
        meta = info.meta()
        all_mids = info.all_mids()
        assets = []
        for coin in meta["universe"]:
            name = coin["name"]
            if name in all_mids:
                assets.append({"symbol": name, "price": float(all_mids[name])})
        return assets
    except Exception as e:
        return []

@app.post("/api/account-status")
def check_status(data: dict):
    # Just check if user exists on chain
    try:
        state = info.user_state(data["wallet_address"])
        exists = state.get("marginSummary", None) is not None
        return {"exists": exists}
    except:
        return {"exists": False}

@app.get("/positions/{address}")
def get_positions(address: str):
    try:
        state = info.user_state(address)
        raw_pos = state.get("assetPositions", [])
        positions = []
        for p in raw_pos:
            data = p["position"]
            size = float(data["szi"])
            if size != 0:
                positions.append({
                    "coin": data["coin"],
                    "size": size,
                    "entry_price": float(data["entryPx"]),
                    "unrealized_pnl": float(data["unrealizedPnl"]),
                    "side": "LONG" if size > 0 else "SHORT"
                })
        
        acct_val = 0
        if "marginSummary" in state:
            acct_val = float(state["marginSummary"].get("accountValue", 0))

        return {"positions": positions, "account_value": {"total_value": acct_val}}
    except:
        return {"positions": [], "account_value": {"total_value": 0}}
