"""
EXCHANGE CLIENT (KuCoin Futures — Professional 2026)
----------------------------------------------------
A safe, robust wrapper for market interaction.

Features:
- DRY_RUN support (no real orders)
- Isolated margin + leverage initialization
- Safe market entry + reduceOnly TP/SL
- Auto-close opposite positions
- Error-tolerant fetching (positions, balance, etc.)
"""

import os
import time
import ccxt
from dotenv import load_dotenv
import config

load_dotenv()


class ExchangeClient:

    def __init__(self):
        self.dry = config.DRY_RUN

        # Initialize CCXT client
        self.exchange = ccxt.kucoinfutures({
            "apiKey": os.getenv("KUCOIN_API_KEY"),
            "secret": os.getenv("KUCOIN_API_SECRET"),
            "password": os.getenv("KUCOIN_PASSPHRASE"),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

        # Load markets
        self.exchange.load_markets()

        # Apply leverage + margin mode to each symbol
        for sym in config.SYMBOLS:
            try:
                self.exchange.set_leverage(config.LEVERAGE, sym)
                self.exchange.set_margin_mode(config.MARGIN_MODE, sym)
            except Exception:
                # Some symbols might reject margin-mode setting → safe to ignore
                pass

    # ───────────────────────────────────────────────
    # Safe Fetching
    # ───────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit=200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def get_balance_usdt(self) -> float:
        try:
            bal = self.exchange.fetch_balance(params={"type": "futures"})
            return float(bal["USDT"]["free"])
        except Exception:
            return 0.0

    def get_position(self, symbol: str):
        try:
            pos = self.exchange.fetch_positions([symbol])
            return pos[0] if pos else None
        except Exception:
            return None

    # ───────────────────────────────────────────────
    # Position conversion helpers
    # ───────────────────────────────────────────────
    def _contract_size(self, symbol: str) -> float:
        return self.exchange.market(symbol)["contractSize"]

    def _to_contracts(self, symbol: str, base_amount: float) -> float:
        cs = self._contract_size(symbol)
        return base_amount / cs

    # ───────────────────────────────────────────────
    # Execution logic (LIVE or DRY RUN)
    # ───────────────────────────────────────────────
    def execute_trade(self, symbol: str, side: str, base_amount: float, sl: float, tp: float):
        """
        side: "buy" or "sell"
        base_amount: amount in base currency (e.g., 0.15 ETH)
        """

        # DRY RUN mode → no real orders
        if self.dry:
            return {
                "dry_run": True,
                "symbol": symbol,
                "side": side,
                "size_base": base_amount,
                "sl": sl,
                "tp": tp,
            }

        contracts = self._to_contracts(symbol, base_amount)
        contracts = self.exchange.amount_to_precision(symbol, contracts)

        # Check for existing position
        pos = self.get_position(symbol)

        if pos:
            direction = pos["side"]  # "long" or "short"
            opposite = "sell" if direction == "long" else "buy"

            if (side == "buy" and direction == "short") or (side == "sell" and direction == "long"):
                # Close opposite position
                try:
                    self.exchange.create_order(
                        symbol, "market", opposite, pos["contracts"], None, {"reduceOnly": True}
                    )
                    time.sleep(0.5)
                except Exception:
                    pass

        # Market entry
        try:
            order = self.exchange.create_order(
                symbol, "market", side, contracts, None,
                {"leverage": config.LEVERAGE}
            )
        except Exception as e:
            return {"error": f"Entry failed: {e}"}

        time.sleep(0.5)

        # Exit side
        exit_side = "sell" if side == "buy" else "buy"

        # Take Profit
        try:
            tp_order = self.exchange.create_order(
                symbol, "limit", exit_side, contracts, tp, {"reduceOnly": True}
            )
        except Exception:
            tp_order = None

        # Stop Loss
        try:
            sl_order = self.exchange.create_order(
                symbol, "market", exit_side, contracts, None,
                {"stop": "loss", "stopPrice": sl, "reduceOnly": True}
            )
        except Exception:
            sl_order = None

        return {
            "entry": order,
            "tp": tp_order,
            "sl": sl_order
        }