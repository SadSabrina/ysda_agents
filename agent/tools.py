"""
The agent's four tools.

All data is hardcoded: no network calls, no costs, reproducible results.
The only source of randomness is the model itself, and that is exactly
what we study.
"""
from __future__ import annotations

WEATHER = {
    "new york": "72°F, sunny with light clouds",
    "london":   "58°F, overcast, light rain",
    "tokyo":    "68°F, clear skies",
    "paris":    "65°F, partly cloudy",
    "berlin":   "55°F, cool and windy",
}

STOCKS = {
    "AAPL":  ("$178.52", "+1.2%"),
    "GOOGL": ("$141.80", "-0.5%"),
    "MSFT":  ("$378.91", "+0.8%"),
    "TSLA":  ("$248.50", "+2.1%"),
    "NVDA":  ("$495.30", "+3.4%"),
}

NEWS = {
    "ai":      "OpenAI released a new model. Google DeepMind announced a breakthrough in reinforcement learning.",
    "finance": "The Fed held rates steady. Markets rose on positive inflation data.",
    "tech":    "Apple unveiled a new MacBook. Microsoft integrated AI into Office.",
}

_ALLOWED_CHARS = set("0123456789+-*/.() ")


def get_weather(args: dict) -> str:
    city = args.get("city", "Unknown")
    return f"Weather in {city}: {WEATHER.get(city.lower(), '70°F, comfortable')}"


def get_stock_price(args: dict) -> str:
    symbol = args.get("symbol", "UNKNOWN").upper()
    price, change = STOCKS.get(symbol, ("$100.00", "0.0%"))
    return f"{symbol}: {price} ({change})"


def calculate(args: dict) -> str:
    expression = args.get("expression", "0")
    if not all(c in _ALLOWED_CHARS for c in expression):
        return "Error: invalid characters"
    try:
        return f"Result: {eval(expression)}"  # noqa: S307 — input is filtered above
    except Exception as exc:
        return f"Calculation error: {exc}"


def get_news(args: dict) -> str:
    topic = args.get("topic", "general")
    return f"News on '{topic}': {NEWS.get(topic.lower(), 'No current news.')}"


_SPECS = [
    (get_weather, "get_weather", "Get the current weather for the given city",
     {"city": ("string", "City name")}),
    (get_stock_price, "get_stock_price", "Get the current stock price by ticker",
     {"symbol": ("string", "Stock ticker, e.g. AAPL")}),
    (calculate, "calculate", "Perform a mathematical calculation",
     {"expression": ("string", "Mathematical expression, e.g. '2 * 3.14 * 5'")}),
    (get_news, "get_news", "Get the latest news on the given topic",
     {"topic": ("string", "News topic: ai, finance, tech")}),
]


def build_tools(create_function_tool) -> list:
    """Build the tools with the TokenSHAP factory.

    create_function_tool comes from load_tokenshap(): the library is loaded
    dynamically, so it cannot be imported at module level.
    """
    out = []
    for executor, name, description, params in _SPECS:
        out.append(create_function_tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {k: {"type": t, "description": d} for k, (t, d) in params.items()},
                "required": list(params),
            },
            executor=executor,
        ))
    return out


TOOL_NAMES = [spec[1] for spec in _SPECS]
