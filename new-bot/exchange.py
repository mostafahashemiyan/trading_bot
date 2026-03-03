import os
import time
from typing import Any, Dict, Optional
from pathlib import Path

import ccxt
from dotenv import load_dotenv

import config

# Load .env
PROJECT_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(PROJECT_ROOT_ENV)


class ExchangeClient:
    """CCXT wrapper for KuCoin Futures (One-Way)."""

    def __init__(self):
        exchange_id = config.EXCHANGE_ID
        api_key = os.getenv("KUCOIN_API_KEY")
        api_secret = os.getenv("KUCOIN_API_SECRET")
        api_password = os.getenv("KUCOIN_API_PASSWORD") or os.getenv("KUCOIN_PASSWORD")

        if not all([api_key, api_secret, api_password]):
            raise ValueError("Missing KuCoin credentials in .env")

        self.exchange = getattr(ccxt, exchange_id)(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )

        self.exchange.load_markets()

        # Set leverage and margin mode (KuCoin is always one-way)
        for symbol in config.SYMBOLS:
            self.exchange.set_leverage(config.LEVERAGE, symbol)
            self.exchange.set_margin_mode(config.MARGIN_MODE, symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str):
        return self.exchange.fetch_ticker(symbol)

    def _market(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.market(symbol)

    def _to_contracts(self, symbol: str, base_amount: float) -> float:
        contract_size = self._market(symbol)['contractSize']
        return base_amount / contract_size  # Correct: contracts = base / size

    def get_balance_usdt(self) -> float:
        bal = self.exchange.fetch_balance(params={'type': 'futures'})
        return float(bal['USDT'].get('free', 0.0))

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        positions = self.exchange.fetch_positions([symbol])
        return positions[0] if positions else None

    def execute_trade(self, symbol: str, side: str, base_amount: float, sl: float, tp: float) -> Dict[str, Any]:
        contracts = self._to_contracts(symbol, base_amount)
        contracts = self.exchange.amount_to_precision(symbol, contracts)

        # Check current position
        pos = self.get_position(symbol)
        current_side = 'buy' if pos and pos['side'] == 'long' else 'sell' if pos and pos['side'] == 'short' else None
        if current_side and current_side != side:
            # Close existing (opposite)
            close_side = 'sell' if current_side == 'buy' else 'buy'
            self.exchange.create_order(symbol, 'market', close_side, pos['contracts'], None, {'reduceOnly': True})
            time.sleep(2)  # Wait for close

        # Entry: market order
        params = {'leverage': config.LEVERAGE}
        order = self.exchange.create_order(symbol, 'market', side, contracts, None, params)
        time.sleep(2)

        # TP/SL
        exit_side = "sell" if side == "buy" else "buy"

        tp_order = None
        try:
            tp_order = self.exchange.create_order(
                symbol, "limit", exit_side, contracts, tp, {"reduceOnly": True}
            )
        except Exception as e:
            print(f"[Exchange] TP failed ({symbol}): {e}")

        sl_order = None
        try:
            sl_order = self.exchange.create_order(
                symbol, "market", exit_side, contracts, None,
                {"stop": "loss", "stopPrice": sl, "reduceOnly": True}
            )
        except Exception as e:
            try:
                sl_order = self.exchange.create_order(
                    symbol, "market", exit_side, contracts, None,
                    {"triggerPrice": sl, "reduceOnly": True, "stop": "loss"}
                )
            except Exception as e2:
                print(f"[Exchange] SL failed ({symbol}): {e} | {e2}")

        return {
            "entry": order,
            "tp": tp_order,
            "sl": sl_order,
        }

    def cancel_orphaned_orders(self, symbol: str, tp_id: Optional[str], sl_id: Optional[str]):
        if not tp_id or not sl_id:
            return

        try:
            tp = self.exchange.fetch_order(tp_id, symbol)
        except Exception:
            tp = None

        try:
            sl = self.exchange.fetch_order(sl_id, symbol)
        except Exception:
            sl = None

        if tp and tp['status'] == 'closed':
            try:
                self.exchange.cancel_order(sl_id, symbol)
            except Exception:
                pass

        if sl and sl['status'] == 'closed':
            try:
                self.exchange.cancel_order(tp_id, symbol)
            except Exception:
                pass