"""
Четыре инструмента агента.

Все данные захардкожены: никуда не ходим, ничего не платим, результат
воспроизводим. Единственный источник стохастичности — сама модель,
и это ровно то, что мы изучаем.
"""
from __future__ import annotations

WEATHER = {
    "new york": "72°F, солнечно с лёгкими облаками",
    "london":   "58°F, пасмурно, лёгкий дождь",
    "tokyo":    "68°F, ясное небо",
    "paris":    "65°F, переменная облачность",
    "berlin":   "55°F, прохладно, ветрено",
}

STOCKS = {
    "AAPL":  ("$178.52", "+1.2%"),
    "GOOGL": ("$141.80", "-0.5%"),
    "MSFT":  ("$378.91", "+0.8%"),
    "TSLA":  ("$248.50", "+2.1%"),
    "NVDA":  ("$495.30", "+3.4%"),
}

NEWS = {
    "ai":      "OpenAI выпустила новую модель. Google DeepMind объявил о прорыве в обучении с подкреплением.",
    "finance": "ФРС сохранила ставку. Рынки выросли на фоне позитивных данных по инфляции.",
    "tech":    "Apple представила новый MacBook. Microsoft интегрировал ИИ в Office.",
}

_ALLOWED_CHARS = set("0123456789+-*/.() ")


def get_weather(args: dict) -> str:
    city = args.get("city", "Unknown")
    return f"Погода в {city}: {WEATHER.get(city.lower(), '70°F, комфортно')}"


def get_stock_price(args: dict) -> str:
    symbol = args.get("symbol", "UNKNOWN").upper()
    price, change = STOCKS.get(symbol, ("$100.00", "0.0%"))
    return f"{symbol}: {price} ({change})"


def calculate(args: dict) -> str:
    expression = args.get("expression", "0")
    if not all(c in _ALLOWED_CHARS for c in expression):
        return "Ошибка: недопустимые символы"
    try:
        return f"Результат: {eval(expression)}"  # noqa: S307 — вход отфильтрован выше
    except Exception as exc:
        return f"Ошибка вычисления: {exc}"


def get_news(args: dict) -> str:
    topic = args.get("topic", "general")
    return f"Новости по теме '{topic}': {NEWS.get(topic.lower(), 'Нет актуальных новостей.')}"


_SPECS = [
    (get_weather, "get_weather", "Получить текущую погоду для указанного города",
     {"city": ("string", "Название города")}),
    (get_stock_price, "get_stock_price", "Получить текущую цену акции по тикеру",
     {"symbol": ("string", "Тикер акции, например AAPL")}),
    (calculate, "calculate", "Выполнить математическое вычисление",
     {"expression": ("string", "Математическое выражение, например '2 * 3.14 * 5'")}),
    (get_news, "get_news", "Получить последние новости по заданной теме",
     {"topic": ("string", "Тема новостей: ai, finance, tech")}),
]


def build_tools(create_function_tool) -> list:
    """Собрать инструменты фабрикой из TokenSHAP.

    create_function_tool приходит из load_tokenshap(): библиотека грузится
    динамически, поэтому импортировать её на уровне модуля нельзя.
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
