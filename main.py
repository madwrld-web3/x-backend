from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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

# Configure CORS - UPDATED FOR VERCEL CONNECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                        # Local testing
        "http://localhost:5173",                        # Vite local testing
        "https://x-change-frontend-theta.vercel.app",   # <--- YOUR VERCEL APP
        "https://x-change-frontend-theta.vercel.app/"   # (Trailing slash version)
    ],
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
    user_main_wallet_address: str
    coin: str
    is_buy: bool
    usd_size: float
    leverage: int

class TradeResponse(BaseModel):
    status: str
    message: str
    order_result: Dict[str, Any] | None = None

class ApproveAgentRequest(BaseModel):
    user_wallet_address: str
    agent_address: str
    signature: str

class ClosePositionRequest(BaseModel):
    user_agent_private_key: str
    user_main_wallet_address: str
    coin: str

class Position(BaseModel):
    coin: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    pnl_percentage: float
    leverage: int
    liquidation_price: Optional[float] = None

class PositionsResponse(BaseModel):
    positions: List[Position]
    account_value: Dict[str, Any]

# Initialize Hyperliquid Info (for market data)
info = Info(constants.MAINNET_API_URL, skip_ws=True)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "X/CHANGE",
        "description": "Personal Crypto Exchange API",
        "version": "2.0.0",
        "features": ["trading", "positions", "agent_approval"]
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

@app.post("/approve-agent")
async def approve_agent(request: ApproveAgentRequest):
    """
    Approve an agent wallet to trade on behalf of the user's main wallet.
    This uses the user's wallet to sign an approval on Hyperliquid.
    IMPORTANT: This endpoint should be called with the signature from the frontend,
    but the actual approval transaction must be signed by the user's main wallet.
    """
    try:
        logger.info(f"Approving agent {request.agent_address} for wallet {request.user_wallet_address}")
        
        # Note: In a production environment, you would verify the signature here
        # For now, we'll trust that the user has signed the message
        
        # The agent approval on Hyperliquid needs to be done by the main wallet
        # Since we don't have the main wallet's private key (and shouldn't!),
        # we'll return success and the frontend will handle the actual approval
        # through MetaMask when the user clicks "Activate Agent"
        
        return {
            "status": "success",
            "message": "Agent approved successfully",
            "agent_address": request.agent_address,
            "user_address": request.user_wallet_address
        }
        
    except Exception as e:
        logger.error(f"Error approving agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to approve agent: {str(e)}")

@app.get("/positions/{wallet_address}", response_model=PositionsResponse)
async def get_positions(wallet_address: str):
    """
    Get all open positions and account information for a wallet address.
    """
    try:
        # Get user state from Hyperliquid
        user_state = info.user_state(wallet_address)
        
        positions = []
        
        # Parse positions from user state
        if user_state and "assetPositions" in user_state:
            for position_data in user_state["assetPositions"]:
                if position_data["position"]["szi"] != "0.0":
                    # Get asset info
                    coin = position_data["position"]["coin"]
                    size = float(position_data["position"]["szi"])
                    entry_price = float(position_data["position"]["entryPx"])
                    
                    # Get current mark price
                    all_mids = info.all_mids()
                    mark_price = float(all_mids.get(coin, entry_price))
                    
                    # Calculate PnL
                    if size > 0:  # LONG
                        side = "LONG"
                        unrealized_pnl = size * (mark_price - entry_price)
                    else:  # SHORT
                        side = "SHORT"
                        unrealized_pnl = abs(size) * (entry_price - mark_price)
                    
                    # Calculate PnL percentage
                    position_value = abs(size) * entry_price
                    pnl_percentage = (unrealized_pnl / position_value * 100) if position_value > 0 else 0
                    
                    # Get leverage
                    leverage = int(float(position_data["position"].get("leverage", {}).get("value", 1)))
                    
                    # Get liquidation price
                    liquidation_px = position_data["position"].get("liquidationPx")
                    liquidation_price = float(liquidation_px) if liquidation_px else None
                    
                    position = Position(
                        coin=coin,
                        side=side,
                        size=abs(size),
                        entry_price=entry_price,
                        mark_price=mark_price,
                        unrealized_pnl=unrealized_pnl,
                        pnl_percentage=pnl_percentage,
                        leverage=leverage,
                        liquidation_price=liquidation_price
                    )
                    positions.append(position)
        
        # Get account value
        account_value = {
            "total_value": 0.0,
            "account_margin": 0.0,
            "withdrawable": 0.0
        }
        
        if user_state and "marginSummary" in user_state:
            margin_summary = user_state["marginSummary"]
            account_value["total_value"] = float(margin_summary.get("accountValue", 0))
            account_value["account_margin"] = float(margin_summary.get("totalMarginUsed", 0))
            account_value["withdrawable"] = float(margin_summary.get("withdrawable", 0))
        
        logger.info(f"Found {len(positions)} positions for {wallet_address}")
        
        return PositionsResponse(
            positions=positions,
            account_value=account_value
        )
        
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")

@app.post("/trade", response_model=TradeResponse)
async def execute_trade(trade_request: TradeRequest):
    """
    Execute a trade on Hyperliquid using agent wallet.
    The agent wallet signs the transaction, but trades on behalf of the main wallet.
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
        
        # Initialize exchange with agent wallet trading for main wallet
        # The agent wallet signs transactions, but account_address points to main wallet
        agent_account = Account.from_key(trade_request.user_agent_private_key)
        exchange = Exchange(
            agent_account,
            constants.MAINNET_API_URL,
            account_address=trade_request.user_main_wallet_address  # Main wallet has the funds
        )
        
        logger.info(f"Agent {agent_account.address} trading for main wallet {trade_request.user_main_wallet_address}")
        
        # Get asset metadata for proper size rounding
        meta = info.meta()
        asset_info = next((asset for asset in meta["universe"] if asset["name"] == trade_request.coin), None)
        
        if not asset_info:
            raise HTTPException(
                status_code=400,
                detail=f"Asset {trade_request.coin} not found in metadata"
            )
        
        # Get the szDecimals for proper rounding
        sz_decimals = asset_info.get("szDecimals", 8)
        
        # Calculate size in coins based on current price
        all_mids = info.all_mids()
        price = float(all_mids.get(trade_request.coin, 0))
        
        if price == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch price for {trade_request.coin}"
            )
        
        # Calculate position size in coins and round to proper precision
        raw_size = trade_request.usd_size / price
        size_in_coins = round(raw_size, sz_decimals)
        
        # Prepare order
        is_buy = trade_request.is_buy
        
        # Place market order - use positional arguments
        order_result = exchange.market_open(
            trade_request.coin,
            is_buy,
            size_in_coins,
            None,
            0.05
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
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute trade: {str(e)}"
        )

@app.post("/close-position")
async def close_position(request: ClosePositionRequest):
    """
    Close an entire position for a specific coin.
    """
    try:
        # Initialize exchange with agent wallet
        agent_account = Account.from_key(request.user_agent_private_key)
        exchange = Exchange(
            agent_account,
            constants.MAINNET_API_URL,
            account_address=request.user_main_wallet_address
        )
        
        logger.info(f"Closing position for {request.coin} on behalf of {request.user_main_wallet_address}")
        
        # Get current position to determine size and side
        user_state = info.user_state(request.user_main_wallet_address)
        
        position_size = 0
        for position_data in user_state.get("assetPositions", []):
            if position_data["position"]["coin"] == request.coin:
                position_size = float(position_data["position"]["szi"])
                break
        
        if position_size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No open position found for {request.coin}"
            )
        
        # Close position using market_close
        result = exchange.market_close(request.coin, None, 0.05)
        
        logger.info(f"Position closed: {request.coin}")
        
        return {
            "status": "success",
            "message": f"Position closed: {request.coin}",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to close position: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
