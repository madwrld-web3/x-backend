from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="X/CHANGE", description="Personal Crypto Exchange API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Asset(BaseModel):
    symbol: str
    name: str
    max_leverage: int
    price: float

class TradeRequest(BaseModel):
    user_agent_private_key: str
    coin: str
    is_buy: bool
    usd_size: float
    leverage: int

class TradeResponse(BaseModel):
    status: str
    message: str
    order_result: Dict[str, Any] | None = None

# Initialize Hyperliquid Info (for market data)
info = Info(constants.MAINNET_API_URL, skip_ws=True)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "X/CHANGE",
        "description": "Personal Crypto Exchange API",
        "version": "1.0.0"
    }

@app.get("/markets", response_model=List[Asset])
async def get_markets():
    """
    Fetch market data from Hyperliquid and return simplified list of assets.
    Includes BTC, ETH, SOL, HYPE, PAXG, and all HIP assets.
    """
    try:
        # Fetch Meta and Asset Context from Hyperliquid
        meta = info.meta()
        all_mids = info.all_mids()
        
        # Target coins to include
        target_coins = {"BTC", "ETH", "SOL", "HYPE", "PAXG"}
        
        assets = []
        
        for asset_info in meta["universe"]:
            symbol = asset_info["name"]
            
            # Include target coins or any HIP assets
            if symbol in target_coins or symbol.startswith("HIP"):
                # Get current price from all_mids
                price = float(all_mids.get(symbol, 0))
                
                # Extract max leverage
                max_leverage = asset_info.get("maxLeverage", 50)
                
                asset = Asset(
                    symbol=symbol,
                    name=symbol,
                    max_leverage=max_leverage,
                    price=price
                )
                assets.append(asset)
        
        # Sort by symbol for consistent ordering
        assets.sort(key=lambda x: x.symbol)
        
        logger.info(f"Fetched {len(assets)} assets from Hyperliquid")
        return assets
        
    except Exception as e:
        logger.error(f"Error fetching markets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@app.post("/trade", response_model=TradeResponse)
async def execute_trade(trade_request: TradeRequest):
    """
    Execute a trade on Hyperliquid.
    
    Hard constraint: Maximum leverage is 10x.
    Calculates 3% fee on USD size and logs it.
    """
    try:
        # Hard constraint: Check leverage
        if trade_request.leverage > 10:
            raise HTTPException(
                status_code=400,
                detail="Max leverage is 10x"
            )
        
        # Validate leverage is positive
        if trade_request.leverage < 1:
            raise HTTPException(
                status_code=400,
                detail="Leverage must be at least 1x"
            )
        
        # Calculate 3% fee
        fee_amount = trade_request.usd_size * 0.03
        fee_address = "0x7505E9d72fc210958ca6A62CD1dcaacC6a41E0D4"
        logger.info(f"Fee of ${fee_amount:.2f} due to {fee_address}")
        
        # Initialize exchange with user's private key
        account = Account.from_key(trade_request.user_agent_private_key)
        exchange = Exchange(
            account,
            constants.MAINNET_API_URL,
            account_address=account.address
        )
        
        # Calculate size in coins based on current price
        all_mids = info.all_mids()
        price = float(all_mids.get(trade_request.coin, 0))
        
        if price == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch price for {trade_request.coin}"
            )
        
        # Calculate position size in coins
        size_in_coins = trade_request.usd_size / price
        
        # Prepare order
        is_buy = trade_request.is_buy
        
        # Place market order
        order_result = exchange.market_open(
            asset=trade_request.coin,
            is_buy=is_buy,
            sz=size_in_coins,
            slippage=0.05  # 5% slippage tolerance
        )
        
        logger.info(f"Trade executed: {trade_request.coin} {'BUY' if is_buy else 'SELL'} "
                   f"${trade_request.usd_size} at {trade_request.leverage}x leverage")
        
        return TradeResponse(
            status="success",
            message=f"Trade executed successfully: {'BUY' if is_buy else 'SELL'} "
                   f"{size_in_coins:.4f} {trade_request.coin} at {trade_request.leverage}x leverage",
            order_result=order_result
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute trade: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
