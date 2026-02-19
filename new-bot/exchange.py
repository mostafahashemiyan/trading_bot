import os
import ccxt
import time
from dotenv import load_dotenv

from config import USE_FUTURES, LEVERAGE, MARGIN_MODE

load_dotenv()


def _make_exchange():
    """Create CCXT exchange instance.

    Note: This bot is SHORT-only, so we default to KuCoin Futures for real shorts.
    """
    common = {
        "apiKey": os.getenv("KUCOIN_API_KEY"),
        "secret": os.getenv("KUCOIN_API_SECRET"),
        "password": os.getenv("KUCOIN_API_PASSPHRASE"),
        "enableRateLimit": True,
    }

    if USE_FUTURES:
        # KuCoin Futures (USDT-margined perpetuals / swaps)
        # CCXT: https://docs.ccxt.com/
        ex = ccxt.kucoinfutures({
            **common,
            "options": {
                "defaultType": "swap",
            },
        })
        return ex

    # Spot (cannot short)
    return ccxt.kucoin(common)


exchange = _make_exchange()


def normalize_symbol(symbol: str) -> str:
    """Best-effort symbol normalizer for KuCoin Futures.

    Users often provide "BTC/USDT" while swaps are "BTC/USDT:USDT".
    """
    try:
        markets = exchange.load_markets()
    except Exception:
        markets = getattr(exchange, "markets", {}) or {}

    if symbol in markets:
        return symbol

    if USE_FUTURES:
        # Try common KuCoin swap symbol format
        if ":" not in symbol and symbol.endswith("/USDT"):
            candidate = symbol + ":USDT"
            if candidate in markets:
                return candidate

    return symbol


def fetch_ohlcv(symbol, timeframe, limit=200):
    symbol = normalize_symbol(symbol)
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


def get_balance(currency="USDT"):
    """Fetch free balance for the quote currency."""
    try:
        bal = exchange.fetch_balance()

        # Futures accounts can report balances under different keys.
        # We try the standard unified format first.
        free = (bal.get("free") or {}).get(currency)
        if free is not None:
            return float(free)

        # Fallbacks
        total = (bal.get("total") or {}).get(currency)
        if total is not None:
            return float(total)

        return 0.0

    except Exception as e:
        print(f"Error fetching balance: {e}")
        return 0.0


def execute_trade(
    symbol, side, entry_price, stop_loss, take_profit, risk_per_trade=0.01
):
    """
    1. Calculates Position Size based on Risk %.
    2. Places Market Entry.
    3. Places Stop Loss Order.
    4. Places Take Profit Order.
    """
    try:
        symbol = normalize_symbol(symbol)

        if (not USE_FUTURES) and side.lower() == "sell":
            print("Refusing to place SHORT on spot. Enable USE_FUTURES in config.py")
            return None

        # 1. Calculate Balance & Size
        quote_currency = "USDT"  # for USDT-margined futures; best-effort for spot too
        balance = get_balance(quote_currency)

        if balance == 0:
            print("Insufficient balance.")
            return None

        # Risk Calculation: (Balance * Risk%) / (Entry - Stop)
        risk_amt = balance * risk_per_trade
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            print("Invalid Stop Loss (Price Diff is 0).")
            return None

        amount = risk_amt / price_diff

        # Sanity check: Kraken min order size (approx)
        if amount * entry_price < 10:
            print("Position size too small (< $10).")
            return None

        # Best-effort leverage/margin mode setup (futures)
        if USE_FUTURES:
            try:
                exchange.set_margin_mode(MARGIN_MODE, symbol)
            except Exception:
                pass
            try:
                exchange.set_leverage(int(LEVERAGE), symbol)
            except Exception:
                pass

        print(f"EXECUTING {side} {symbol}")
        print(f"Balance: {balance:.2f} | Risk: ${risk_amt:.2f} | Size: {amount:.4f}")

        # 2. Execute Market Entry
        order = exchange.create_order(symbol, "market", side, amount)
        print(f"Entry Filled: {order['id']}")

        # 3. Place Stop Loss (Wait a moment to ensure entry is processed)
        time.sleep(2)

        # Determine SL/TP side (Opposite of entry)
        exit_side = "sell" if side == "buy" else "buy"

        # 3. Place Stop Loss + 4. Take Profit
        # KuCoin Futures uses conditional orders with params stop/stopPrice.
        # CCXT docs/FAQ: stop orders are exchange-specific. We use best-effort params.
        # - Stop loss for SHORT triggers when price goes UP.
        # - Take profit for SHORT triggers when price goes DOWN.
        if USE_FUTURES:
            stop_direction = "up" if side == "sell" else "down"
            sl_params = {
                "stop": stop_direction,
                "stopPrice": float(stop_loss),
                "reduceOnly": True,
            }
            # Some KuCoin endpoints support closeOrder to ensure position-reducing.
            sl_params["closeOrder"] = True

            sl_order = exchange.create_order(symbol, "market", exit_side, amount, params=sl_params)
            print(f"SL (stop-market) placed at {stop_loss}: {sl_order.get('id')}")

            tp_order = exchange.create_order(
                symbol,
                "limit",
                exit_side,
                amount,
                price=float(take_profit),
                params={"reduceOnly": True},
            )
            print(f"TP (reduce-only limit) placed at {take_profit}: {tp_order.get('id')}")
        else:
            # Spot fallback (LONG only).
            sl_order = None
            tp_order = exchange.create_order(
                symbol,
                "limit",
                exit_side,
                amount,
                price=float(take_profit),
            )
            print(f"TP placed at {take_profit}: {tp_order.get('id')}")

        return {
            "entry_id": order["id"],
            "sl_id": sl_order["id"] if sl_order else None,
            "tp_id": tp_order["id"] if tp_order else None,
            "amount": amount,
            "entry_price": entry_price,
        }

    except Exception as e:
        print(f"Execution Failed: {e}")
        return None
