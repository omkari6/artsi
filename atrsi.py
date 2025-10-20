import os
import time
import math
import traceback
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# ==== Config ====
@dataclass
class Config:
    symbol: str = "SPXUSDT"    # e.g., BTCUSDT, SPXUSDT
    interval: str = "5m"        # Your requested default
    rsi_period: int = 14
    rsi_high: float = 52.5      # Your requested default
    rsi_low: float = 42.5       # Your requested default
    atr_period: int = 14
    atr_multiple: float = 0.95  # From your original

    # Sizing
    use_fixed_notional: bool = True
    fixed_usdt_notional: float = 100.0  # Your requested default
    fixed_quantity: float = 1.0

    leverage: int = 5
    poll_interval_sec: float = 1.0

    # Bar-open detection grace (seconds from exact open)
    entry_open_grace_sec: float = 10.0

    # Exit order behavior
    sl_use_stop_market: bool = False
    stop_limit_ticks: int = 3

    # Bracket anchoring and safety
    min_buffer_ticks: int = 2

    # Restored: Hold for exactly this many intervals (1 = original one-bar trade)
    hold_intervals: int = 1  # Set via env var, e.g., export BOT_HOLD_INTERVALS=2 for 10m on 5m

    # Logging
    verbose: bool = True


# ==== Utils ====
def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    qty = int(interval[:-1])
    if unit == 'm':
        return qty * 60_000
    if unit == 'h':
        return qty * 60 * 60_000
    if unit == 'd':
        return qty * 24 * 60 * 60_000
    raise ValueError(f"Unsupported interval: {interval}")


def round_step(value: float, step: float) -> float:
    if step == 0:
        return value
    return float(math.floor(value / step) * step)


def clamp_tick(price: float, tick: float) -> float:
    if tick == 0:
        return price
    precision = max(0, int(round(-math.log10(tick))))
    return float(round(price, precision))


# ==== Retry Wrapper for API Calls ====
def with_retry(func, max_retries=3, base_sleep=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except BinanceAPIException as e:
            if e.code == -1007:  # Timeout
                sleep_time = base_sleep * (2 ** attempt)
                print(f"[RETRY] API timeout (attempt {attempt+1}/{max_retries}), sleeping {sleep_time}s")
                time.sleep(sleep_time)
                continue
            raise
    raise RuntimeError(f"[ERROR] Max retries exceeded for API call")


# ==== Binance helpers ====
def futures_filters(client: Client, symbol: str) -> Dict[str, float]:
    def _call():
        info = client.futures_exchange_info()
        tick = 0.0
        step = 0.0
        min_qty = 0.0
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                    elif f["filterType"] in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                        step = float(f["stepSize"])
                        min_qty = float(f["minQty"])
                break
        return {"tick": tick, "step": step, "min_qty": min_qty}
    return with_retry(_call)


def server_time_ms(client: Client) -> int:
    def _call():
        try:
            return int(client.futures_time()["serverTime"])
        except Exception:
            return int(client.get_server_time()["serverTime"])
    return with_retry(_call)


def get_klines(client: Client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    def _call():
        raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        cols = [
            "open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades",
            "taker_base_vol", "taker_quote_vol", "ignore"
        ]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "qav", "taker_base_vol", "taker_quote_vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df
    return with_retry(_call)


def mark_price(client: Client, symbol: str) -> float:
    def _call():
        return float(client.futures_mark_price(symbol=symbol)["markPrice"])
    return with_retry(_call)


def position_amt(client: Client, symbol: str) -> float:
    def _call():
        pos = client.futures_position_information(symbol=symbol)
        if not pos:
            return 0.0
        return float(pos[0]["positionAmt"])
    return with_retry(_call)


def market_order(client: Client, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
    qty = max(qty, 0.0)
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": str(qty),
        "reduceOnly": "true" if reduce_only else "false",
        "newOrderRespType": "RESULT",
    }
    def _call():
        return client.futures_create_order(**params)
    return with_retry(_call)


def change_leverage_oneway(client: Client, symbol: str, leverage: int):
    def _get_mode():
        return client.futures_get_position_mode()["dualSidePosition"]
    
    try:
        current_mode = with_retry(_get_mode)
        if current_mode:
            def _change_mode():
                client.futures_change_position_mode(dualSidePosition="false")
            with_retry(_change_mode)
            print("[LOG] Changed to one-way position mode")
        else:
            print("[LOG] Already in one-way position mode - no change needed")
    except Exception as e:
        print(f"[WARN] Failed to check/set position mode: {e} - Continuing")

    try:
        def _change_leverage():
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
        with_retry(_change_leverage)
        print(f"[LOG] Leverage set to {leverage}x")
    except Exception as e:
        print(f"[WARN] Failed to set leverage: {e} - Continuing")


def cancel_all_open_orders(client: Client, symbol: str):
    def _call():
        client.futures_cancel_all_open_orders(symbol=symbol)
    try:
        with_retry(_call)
    except BinanceAPIException as e:
        if e.code != -2011:
            raise


# ==== Indicators ====
def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    return RSIIndicator(close=close, window=period).rsi()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


# ==== Strategy ====
class RSIAtrBot:
    def __init__(self, client: Client, cfg: Config):
        self.client = client
        self.cfg = cfg
        f = futures_filters(client, cfg.symbol)
        self.tick = f["tick"]
        self.step = f["step"]
        self.min_qty = f["min_qty"]
        change_leverage_oneway(client, cfg.symbol, cfg.leverage)

        self.interval_ms = interval_to_ms(cfg.interval)
        self.last_seen_open_ms: Optional[int] = None

        # Position state (restored original time-stop fields)
        self.in_position = False
        self.side: Optional[str] = None
        self.entry_fill: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None
        self.bar_open_ms: Optional[int] = None
        self.bar_close_ms: Optional[int] = None

    def log(self, msg: str):
        if self.cfg.verbose:
            print(msg, flush=True)

    def compute_qty(self, current: float) -> float:
        if not self.cfg.use_fixed_notional:
            qty = self.cfg.fixed_quantity
        else:
            qty = self.cfg.fixed_usdt_notional / max(current, 1e-12)
        qty = round_step(qty, self.step)
        if qty < self.min_qty:
            self.log(f"[SKIP] Calculated qty={qty} below min_qty={self.min_qty} - adjust notional")
            return 0.0
        return qty

    def place_exit_orders(self, side: str, tp_price: float, sl_trigger: float):
        cancel_all_open_orders(self.client, self.cfg.symbol)
        self.log("[LOG] Cancelled all open orders before placing exits")

        qty = abs(position_amt(self.client, self.cfg.symbol))
        if qty <= 0:
            self.log("[WARN] No position qty for exits, skipping")
            return

        tp_side = "SELL" if side == "LONG" else "BUY"
        tp_price = clamp_tick(tp_price, self.tick)
        def _tp_call():
            return self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side=tp_side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=str(qty),
                price=f"{tp_price:.10f}",
                reduceOnly="true",
                newOrderRespType="RESULT"
            )
        tp = with_retry(_tp_call)
        self.log(f"[EXIT ORDERS] TP LIMIT placed #{tp.get('orderId')} @ {tp_price}")

        sl_side = "SELL" if side == "LONG" else "BUY"
        if self.cfg.sl_use_stop_market:
            def _sl_market_call():
                return self.client.futures_create_order(
                    symbol=self.cfg.symbol,
                    side=sl_side,
                    type="STOP_MARKET",
                    stopPrice=f"{sl_trigger:.10f}",
                    closePosition="true",
                    workingType="MARK_PRICE",
                    newOrderRespType="RESULT"
                )
            sl = with_retry(_sl_market_call)
            self.log(f"[EXIT ORDERS] SL STOP-MARKET placed trigger @ {sl_trigger}")
        else:
            tick = self.tick if self.tick > 0 else 0.0
            offset = self.cfg.stop_limit_ticks * tick
            if side == "LONG":
                sl_limit = clamp_tick(max(sl_trigger - offset, 0.0), tick) if tick > 0 else sl_trigger
            else:
                sl_limit = clamp_tick(sl_trigger + offset, tick) if tick > 0 else sl_trigger
            def _sl_limit_call():
                return self.client.futures_create_order(
                    symbol=self.cfg.symbol,
                    side=sl_side,
                    type="STOP",
                    timeInForce="GTC",
                    quantity=str(qty),
                    price=f"{sl_limit:.10f}",
                    stopPrice=f"{sl_trigger:.10f}",
                    reduceOnly="true",
                    workingType="MARK_PRICE",
                    newOrderRespType="RESULT"
                )
            sl = with_retry(_sl_limit_call)
            self.log(f"[EXIT ORDERS] SL STOP-LIMIT placed trigger @ {sl_trigger} limit @ {sl_limit}")

    def detect_bar_open(self) -> Optional[int]:
        now_ms = server_time_ms(self.client)
        open_ms = (now_ms // self.interval_ms) * self.interval_ms
        age_ms = now_ms - open_ms
        if self.last_seen_open_ms == open_ms:
            return None
        if age_ms <= int(self.cfg.entry_open_grace_sec * 1000):
            self.last_seen_open_ms = open_ms
            self.log(f"[LOG] Detected new bar open at {open_ms}")
            return open_ms
        return None

    def on_bar_open(self, bar_open_ms: int):
        lookback = max(self.cfg.rsi_period, self.cfg.atr_period) + 5
        df = get_klines(self.client, self.cfg.symbol, self.cfg.interval, limit=lookback)
        if len(df) < lookback - 2:
            self.log("[SKIP] Insufficient klines for indicators")
            return

        last_closed = df.iloc[-2]
        forming = df.iloc[-1]
        ref_open = float(forming["open"])

        df_sig = df.iloc[:-1].copy()
        rsi = calc_rsi(df_sig["close"], self.cfg.rsi_period)
        atr = calc_atr(df_sig["high"], df_sig["low"], df_sig["close"], self.cfg.atr_period)
        rsi_last = float(rsi.iloc[-1])
        atr_last = float(atr.iloc[-1])

        self.log(f"[BAR OPEN] {self.cfg.symbol} {self.cfg.interval} | rsi={rsi_last:.2f} hi={self.cfg.rsi_high} lo={self.cfg.rsi_low} | atr={atr_last:.8f} ref_open={ref_open:.8f}")

        if self.in_position:
            self.log("[SKIP] Already in position - waiting for time-stop")
            return

        # Original signal logic
        side = None
        if rsi_last > self.cfg.rsi_high:
            side = "LONG"
        elif rsi_last < self.cfg.rsi_low:
            side = "SHORT"
        if side is None:
            self.log("[NO SIGNAL] RSI within band")
            return

        current = mark_price(self.client, self.cfg.symbol)
        qty = self.compute_qty(current)
        if qty <= 0:
            self.log(f"[SKIP] Invalid qty={qty} for entry - check notional/price/step")
            return

        entry_side = "BUY" if side == "LONG" else "SELL"
        self.log(f"[ENTRY] {side} qty={qty:.2f} (notional={self.cfg.fixed_usdt_notional:.2f} / price={current:.4f}) | ref_open={ref_open:.8f} current={current:.8f}")
        try:
            entry_res = market_order(self.client, self.cfg.symbol, entry_side, qty=qty, reduce_only=False)
            self.log(f"[ENTRY] Result: {entry_res}")
        except BinanceAPIException as e:
            self.log(f"[ENTRY FAILED] {e}")
            return

        entry_px = float(entry_res.get("avgPrice") or current)
        dist = self.cfg.atr_multiple * atr_last
        tick = self.tick if self.tick > 0 else 0.0
        buffer = (self.cfg.min_buffer_ticks * tick) if tick > 0 else 0.0

        if side == "LONG":
            tp = clamp_tick(max(entry_px + dist, entry_px + buffer), tick)
            sl = clamp_tick(min(entry_px - dist, entry_px - buffer), tick)
        else:
            tp = clamp_tick(min(entry_px - dist, entry_px - buffer), tick)
            sl = clamp_tick(max(entry_px + dist, entry_px + buffer), tick)

        try:
            self.place_exit_orders(side, tp, sl)
        except BinanceAPIException as e:
            self.log(f"[EXIT ORDERS FAILED] {e} - Fail-safe closing position")
            self.close_position_market()
            return

        # Set state with restored time-stop
        self.in_position = True
        self.side = side
        self.entry_fill = entry_px
        self.tp = tp
        self.sl = sl
        self.bar_open_ms = bar_open_ms
        self.bar_close_ms = bar_open_ms + (self.interval_ms * self.cfg.hold_intervals)
        self.log(f"[STATE] In {side} | entry_fill={entry_px:.10f} TP={tp:.10f} SL={sl:.10f} | close at {self.bar_close_ms} (after {self.cfg.hold_intervals} interval(s))")

    def close_position_market(self):
        amt = position_amt(self.client, self.cfg.symbol)
        if abs(amt) < 1e-12:
            self.log("[LOG] No position to close")
            return None
        side = "SELL" if amt > 0 else "BUY"
        qty = round_step(abs(amt), self.step)
        if qty <= 0:
            self.log("[LOG] Qty rounded to 0, skipping close")
            return None
        self.log(f"[CLOSE] Market close: side={side} qty={qty}")
        return market_order(self.client, self.cfg.symbol, side, qty=qty, reduce_only=True)

    def manage_position(self):
        if not self.in_position:
            return
        # Check if TP/SL already closed it
        amt = position_amt(self.client, self.cfg.symbol)
        if abs(amt) < 1e-12:
            self.log("[EXIT] Position closed via TP/SL - cancelling residuals")
            cancel_all_open_orders(self.client, self.cfg.symbol)
            self._reset_pos()
            return

        # Restored original time-stop: Close if bar close reached
        now = server_time_ms(self.client)
        if now >= self.bar_close_ms:
            self.log("[EXIT] Time-stop reached (end of interval) - closing at market")
            cancel_all_open_orders(self.client, self.cfg.symbol)
            res = self.close_position_market()
            self.log(f"[EXIT] Close result: {res}")
            self._reset_pos()
            # Re-check for entry if still in current bar's grace period after closure
            now_after_close = server_time_ms(self.client)
            current_open_ms = (now_after_close // self.interval_ms) * self.interval_ms
            age_after_close = now_after_close - current_open_ms
            if age_after_close <= int(self.cfg.entry_open_grace_sec * 1000) and self.last_seen_open_ms == current_open_ms:
                self.log("[LOG] Re-attempting entry after time-stop closure in same bar")
                self.on_bar_open(current_open_ms)

    def _reset_pos(self):
        self.in_position = False
        self.side = None
        self.entry_fill = None
        self.tp = None
        self.sl = None
        self.bar_open_ms = None
        self.bar_close_ms = None
        self.log("[STATE] Position reset")

    def run(self):
        self.log(f"[START] Bot running | {self.cfg.symbol} {self.cfg.interval} | RSI({self.cfg.rsi_period}) > {self.cfg.rsi_high} / < {self.cfg.rsi_low} | ATR({self.cfg.atr_period}) x {self.cfg.atr_multiple}")
        self.log(f"[CONFIG] Effective settings: notional={self.cfg.fixed_usdt_notional} USDT, leverage={self.cfg.leverage}x, poll={self.cfg.poll_interval_sec}s, hold_intervals={self.cfg.hold_intervals}")
        while True:
            try:
                open_ms = self.detect_bar_open()
                self.manage_position()
                if open_ms is not None:
                    self.on_bar_open(open_ms)
                time.sleep(self.cfg.poll_interval_sec)
            except KeyboardInterrupt:
                self.log("[STOP] Interrupted by user")
                break
            except Exception as e:
                self.log(f"[ERROR] Main loop error: {e}")
                traceback.print_exc()
                time.sleep(2.0)


def main():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Please set BINANCE_API_KEY and BINANCE_API_SECRET.")

    cfg = Config(
        symbol=os.getenv("BOT_SYMBOL", "SPXUSDT"),
        interval=os.getenv("BOT_INTERVAL", "1h"),
        rsi_period=int(os.getenv("BOT_RSI_PERIOD", "30")),
        rsi_high=float(os.getenv("BOT_RSI_HIGH", "57.5")),
        rsi_low=float(os.getenv("BOT_RSI_LOW", "42.5")),
        atr_period=int(os.getenv("BOT_ATR_PERIOD", "30")),
        atr_multiple=float(os.getenv("BOT_ATR_MULTIPLE", "0.95")),
        use_fixed_notional=(os.getenv("BOT_USE_FIXED_NOTIONAL", "true").lower() == "true"),
        fixed_usdt_notional=float(os.getenv("BOT_FIXED_USDT_NOTIONAL", "100")),
        fixed_quantity=float(os.getenv("BOT_FIXED_QUANTITY", "1")),
        leverage=int(os.getenv("BOT_LEVERAGE", "5")),
        poll_interval_sec=float(os.getenv("BOT_POLL_INTERVAL_SEC", "1.0")),
        entry_open_grace_sec=float(os.getenv("BOT_ENTRY_OPEN_GRACE_SEC", "10.0")),
        sl_use_stop_market=(os.getenv("BOT_SL_USE_STOP_MARKET", "false").lower() == "true"),
        stop_limit_ticks=int(os.getenv("BOT_STOP_LIMIT_TICKS", "3")),
        min_buffer_ticks=int(os.getenv("BOT_MIN_BUFFER_TICKS", "2")),
        hold_intervals=int(os.getenv("BOT_HOLD_INTERVALS", "1")),  # New for flexibility
        verbose=(os.getenv("BOT_VERBOSE", "true").lower() == "true"),
    )

    client = Client(api_key, api_secret)
    bot = RSIAtrBot(client, cfg)
    bot.run()


if __name__ == "__main__":
    main()
