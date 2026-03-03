import os
import time
from typing import Any, Dict, Optional
from pathlib import Path

import ccxt
from dotenv import load_dotenv

import config

# Ensure we load the project-level .env (one directory above this file)
PROJECT_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(PROJECT_ROOT_ENV)


class ExchangeClient:
    """CCXT wrapper for KuCoin Futures (One-Way).

    Responsibilities:
    - Fetch market data (OHLCV)
    - Read current position
    - Place entry orders
    - Place TP (reduce-only limit) and SL (reduce-only stop-market) orders
    - Handle one-way position flips (close then reopen)

    Notes on One-Way:
    - You can only have one net position per symbol. To go from LONG->SHORT (or reverse),
      you must close the existing position first.
    """

    def __init__(self):
        exchange_id = getattr(config, "EXCHANGE_ID", "kucoinfutures")
        if exchange_id != "kucoinfutures":
            # Safety: user explicitly wants KuCoin Futures.
            exchange_id = "kucoinfutures"

        api_key = os.getenv("KUCOIN_API_KEY")
        api_secret = os.getenv("KUCOIN_API_SECRET")
        # Support both KUCOIN_API_PASSWORD (documented) and KUCOIN_PASSWORD (your current .env)
        api_password = os.getenv("KUCOIN_API_PASSWORD") or os.getenv("KUCOIN_PASSWORD")

        if not api_key or not api_secret or not api_password:
            raise ValueError(
                "Missing KuCoin credentials in .env: KUCOIN_API_KEY / KUCOIN_API_SECRET / KUCOIN_API_PASSWORD"
            )

        self.exchange = getattr(ccxt, exchange_id)(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "options": {
                    # KuCoin Futures (swap) unified behavior
                    "defaultType": "swap",
                },
            }
        )

        # Prime markets for precision/contractSize
        self.exchange.load_markets()

    # ----------------------
    # Data
    # ----------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str):
        return self.exchange.fetch_ticker(symbol)

    # ----------------------
    # Futures helpers
    # ----------------------
    def _market(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.market(symbol)

    def _contract_size(self, symbol: str) -> float:
        m = self._market(symbol)
        # CCXT uses contractSize for derivatives when available
        cs = m.get("contractSize")
        try:
            return float(cs) if cs else 1.0
        except Exception:
            return 1.0

    def _to_contracts(self, symbol: str, base_amount: float) -> float:
        """Convert base-asset amount to contracts for futures markets."""
        cs = self._contract_size(symbol)
        contracts = base_amount / cs if cs else base_amount
        # Respect amount precision if available
        try:
            return float(self.exchange.amount_to_precision(symbol, contracts))
        except Exception:
            return float(contracts)

    def _from_contracts(self, symbol: str, contracts: float) -> float:
        cs = self._contract_size(symbol)
        return float(contracts) * float(cs)

    def _set_leverage_and_margin(self, symbol: str):
        leverage = getattr(config, "LEVERAGE", 1)
        margin_mode = getattr(config, "MARGIN_MODE", "isolated")

        # Some ccxt exchanges require set_margin_mode before leverage.
        try:
            if hasattr(self.exchange, "set_margin_mode"):
                self.exchange.set_margin_mode(margin_mode, symbol, params={})
        except Exception:
            # Not fatal; KuCoin Futures supports it but ccxt versions vary
            pass

        try:
            if hasattr(self.exchange, "set_leverage"):
                self.exchange.set_leverage(leverage, symbol, params={})
        except Exception:
            pass

    # ----------------------
    # Account / Positions
    # ----------------------
    def get_balance_usdt(self) -> float:
        bal = self.exchange.fetch_balance()
        # For USDT margined futures, USDT is the collateral currency
        usdt = bal.get("USDT") or {}
        free = usdt.get("free")
        if free is None:
            # fallback
            free = bal.get("free", {}).get("USDT", 0)
        return float(free or 0)

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return normalized position dict or None.

        Normalized format:
        {
          'symbol': str,
          'side': 'long'|'short',
          'contracts': float,
        }
        """
        # ccxt unified: fetch_positions
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception:
            try:
                positions = self.exchange.fetch_positions()
            except Exception:
                return None

        for p in positions or []:
            if p.get("symbol") != symbol:
                continue
            contracts = p.get("contracts")
            if contracts is None:
                # some versions use 'contractSize'+'size' or 'info'
                contracts = p.get("size")
            try:
                contracts_f = float(contracts or 0)
            except Exception:
                contracts_f = 0.0

            if contracts_f == 0:
                continue

            side = p.get("side")
            if not side:
                # infer from signed contracts if present
                signed = p.get("contracts")
                try:
                    signed_f = float(signed)
                    side = "long" if signed_f > 0 else "short"
                except Exception:
                    side = None

            if side not in ("long", "short"):
                # some exchanges use 'buy'/'sell'
                if side == "buy":
                    side = "long"
                elif side == "sell":
                    side = "short"

            if side not in ("long", "short"):
                # last resort: look into info
                info = p.get("info") or {}
                s = info.get("side") or info.get("positionSide")
                if s in ("long", "short"):
                    side = s

            if side not in ("long", "short"):
                return None

            return {"symbol": symbol, "side": side, "contracts": abs(contracts_f)}

        return None

    def cancel_all_orders(self, symbol: str):
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception:
            # Older versions may not support it
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                for o in open_orders:
                    try:
                        self.exchange.cancel_order(o["id"], symbol)
                    except Exception:
                        pass
            except Exception:
                pass

    def close_position_market(self, symbol: str) -> bool:
        pos = self.get_position(symbol)
        if not pos:
            return True

        contracts = float(pos["contracts"])
        if contracts <= 0:
            return True

        exit_side = "sell" if pos["side"] == "long" else "buy"

        try:
            self.exchange.create_order(
                symbol,
                "market",
                exit_side,
                contracts,
                None,
                {"reduceOnly": True},
            )
            return True
        except Exception as e:
            print(f"[Exchange] close_position_market failed for {symbol}: {e}")
            return False

    # ----------------------
    # Trading
    # ----------------------
    def execute_trade(self, symbol: str, side: str, amount_base: float, sl: float, tp: float):
        """Execute futures trade in One-Way mode.

        Args:
            symbol: CCXT futures symbol (e.g., ETH/USDT:USDT)
            side: 'buy' (open LONG) or 'sell' (open SHORT)
            amount_base: size in base units (e.g. ETH). Converted to contracts internally.
            sl: stop-loss trigger price
            tp: take-profit limit price
        """
        self._set_leverage_and_margin(symbol)

        desired_pos_side = "long" if side == "buy" else "short"

        # One-Way handling: if opposite position exists, close it first.
        current = self.get_position(symbol)
        if current:
            if current["side"] == desired_pos_side:
                # Already in the same direction -> do nothing
                print(f"[Exchange] Position already {current['side']} on {symbol}; skipping new entry.")
                return
            else:
                print(f"[Exchange] Flipping {symbol}: {current['side']} -> {desired_pos_side}")
                self.cancel_all_orders(symbol)
                ok = self.close_position_market(symbol)
                if not ok:
                    print(f"[Exchange] Flip aborted; could not close existing position on {symbol}.")
                    return
                # Give exchange a moment to update position state
                time.sleep(0.5)

        # Convert base amount to contracts
        contracts = self._to_contracts(symbol, amount_base)
        if contracts <= 0:
            raise ValueError(f"Computed contracts <= 0 for {symbol}. base_amount={amount_base}")

        # Entry
        order = self.exchange.create_order(symbol, "market", side, contracts)

        # Place TP (reduce-only limit) + SL (reduce-only stop-market)
        # Exit side is opposite of entry
        exit_side = "sell" if side == "buy" else "buy"

        # Take Profit: reduce-only LIMIT at tp
        tp_order = None
        try:
            tp_order = self.exchange.create_order(
                symbol,
                "limit",
                exit_side,
                contracts,
                tp,
                {"reduceOnly": True},
            )
        except Exception as e:
            print(f"[Exchange] TP order failed ({symbol}): {e}")

        # Stop Loss: reduce-only STOP-MARKET triggered at sl
        sl_order = None
        try:
            sl_order = self.exchange.create_order(
                symbol,
                "market",
                exit_side,
                contracts,
                None,
                {"stopPrice": sl, "reduceOnly": True, "stop": "loss"},
            )
        except Exception as e:
            # ccxt/kucoinfutures sometimes expects a different param key for trigger
            # try a fallback that many derivatives exchanges accept
            try:
                sl_order = self.exchange.create_order(
                    symbol,
                    "market",
                    exit_side,
                    contracts,
                    None,
                    {"triggerPrice": sl, "reduceOnly": True, "stop": "loss"},
                )
            except Exception as e2:
                print(f"[Exchange] SL order failed ({symbol}): {e} | fallback: {e2}")

        return {
            "entry": order,
            "tp": tp_order,
            "sl": sl_order,
        }

    def cancel_orphaned_orders(self, symbol: str, tp_id: Optional[str], sl_id: Optional[str]):
        """If one protective order fills, cancel the other."""
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

        tp_closed = tp and tp.get("status") in ("closed", "canceled")
        sl_closed = sl and sl.get("status") in ("closed", "canceled")

        if tp and tp.get("status") == "closed":
            try:
                self.exchange.cancel_order(sl_id, symbol)
            except Exception:
                pass

        if sl and sl.get("status") == "closed":
            try:
                self.exchange.cancel_order(tp_id, symbol)
            except Exception:
                pass
