from langchain.tools import tool


# Define tools using @tool decorator
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location: The city and state, e.g., 'San Francisco, CA'
    """
    # Simulated weather data
    weather_data = {
        "San Francisco, CA": "Sunny, 72°F",
        "New York, NY": "Cloudy, 65°F",
        "London, UK": "Rainy, 55°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    """
    # Simulated stock data
    stocks = {
        "AAPL": {"price": 178.50, "change": +2.30, "percent_change": +1.31},
        "GOOGL": {"price": 142.80, "change": -1.20, "percent_change": -0.83},
        "MSFT": {"price": 378.91, "change": +5.15, "percent_change": +1.38},
        "TSLA": {"price": 242.84, "change": -3.22, "percent_change": -1.31},
    }

    ticker = ticker.upper()
    if ticker in stocks:
        return {"ticker": ticker, "success": True, **stocks[ticker]}
    else:
        return {"ticker": ticker, "success": False, "error": "Ticker not found"}
