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


# ================= Config =================
@dataclass
class Config:
    symbol: str = "SPXUSDT"          # e.g., BTCUSDT, SPXUSDT
    interval: str = "5m"             # e.g., 5m, 15m, 1h, 4h
    rsi_period: int = 30
    rsi_high: float = 60.0           # long if RSI > rsi_high
    rsi_low: float = 40.0            # short if RSI < rsi_low
    atr_period: int = 30
    atr_multiple: float = 0.95       # TP/SL distance = multiple * ATR

    use_fixed_notional: bool = True
    fixed_usdt_notional: float = 50.0
    fixed_quantity: float = 1.0

    leverage: int = 5
    poll_interval_sec: float = 0.5   # faster poll helps catch bar open easily

    # Bar-open detection grace (seconds from exact open)
    entry_open_grace_sec: float = 10.0

    # SL behavior: STOP-LIMIT (reduce-only) by default (cheaper), or STOP-MARKET (safer fills)
    sl_use_stop_market: bool = False
    stop_limit_ticks: int = 3         # ticks offset for SL limit price

    # Logging
    verbose: bool = True


# ================= Utils =================
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


# ================= Binance helpers =================
def futures_filters(client: Client, symbol: str) -> Dict[str, float]:
    info = client.futures_exchange_info()
    tick = 0.0
    step = 0.0
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                elif f["filterType"] in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                    step = float(f["stepSize"])
            break
    return {"tick": tick, "step": step}


def server_time_ms(client: Client) -> int:
    # futures_time may not exist on older libs; fallback to get_server_time
    try:
        return int(client.futures_time()["serverTime"])
    except Exception:
        return int(client.get_server_time()["serverTime"])


def get_klines(client: Client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    cols = [
        "open_time","open","high","low","close","volume","close_time","qav","num_trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","qav","taker_base_vol","taker_quote_vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def mark_price(client: Client, symbol: str) -> float:
    return float(client.futures_mark_price(symbol=symbol)["markPrice"])


def position_amt(client: Client, symbol: str) -> float:
    pos = client.futures_position_information(symbol=symbol)
    if not pos:
        return 0.0
    return float(pos[0]["positionAmt"])


def market_order(client: Client, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
    return client.futures_create_order(
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=str(qty),
        reduceOnly="true" if reduce_only else "false",
        newOrderRespType="RESULT",
    )


def change_leverage_oneway(client: Client, symbol: str, leverage: int):
    try:
        client.futures_change_position_mode(dualSidePosition="false")
    except Exception:
        pass
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception:
        pass


def cancel_all_open_orders(client: Client, symbol: str):
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
    except BinanceAPIException as e:
        if e.code != -2011:
            raise


# ================= Indicators =================
def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    return RSIIndicator(close=close, window=period).rsi()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


# ================= Strategy =================
class RSIAtrBot:
    def __init__(self, client: Client, cfg: Config):
        self.client = client
        self.cfg = cfg
        f = futures_filters(client, cfg.symbol)
        self.tick = f["tick"]
        self.step = f["step"]
        change_leverage_oneway(client, cfg.symbol, cfg.leverage)

        self.interval_ms = interval_to_ms(cfg.interval)
        self.last_seen_open_ms: Optional[int] = None

        # Position state
        self.in_position = False
        self.side = None
        self.entry_open = None
        self.tp = None
        self.sl = None
        self.bar_open_ms = None
        self.bar_close_ms = None

    def log(self, msg: str):
        if self.cfg.verbose:
            print(msg, flush=True)

    def compute_qty(self, ref_price: float) -> float:
        if self.cfg.use_fixed_notional:
            qty = self.cfg.fixed_usdt_notional / max(ref_price, 1e-12)
        else:
            qty = self.cfg.fixed_quantity
        qty = round_step(qty, self.step)
        return max(qty, 0.0)

    def place_exit_orders(self, side: str, qty: float, tp_price: float, sl_trigger: float):
        cancel_all_open_orders(self.client, self.cfg.symbol)

        # TP LIMIT (reduce-only)
        tp_side = "SELL" if side == "LONG" else "BUY"
        tp_price = clamp_tick(tp_price, self.tick)
        tp = self.client.futures_create_order(
            symbol=self.cfg.symbol,
            side=tp_side,
            type="LIMIT",
            timeInForce="GTC",
            quantity=str(qty),
            price=f"{tp_price:.10f}",
            reduceOnly="true",
            newOrderRespType="RESULT"
        )
        self.log(f"[EXIT ORDERS] TP LIMIT placed #{tp.get('orderId')} @ {tp_price}")

        # SL: STOP-MARKET or STOP (stop-limit)
        sl_side = "SELL" if side == "LONG" else "BUY"
        if self.cfg.sl_use_stop_market:
            sl = self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side=sl_side,
                type="STOP_MARKET",
                stopPrice=f"{sl_trigger:.10f}",
                closePosition="true",
                workingType="MARK_PRICE",
                newOrderRespType="RESULT"
            )
            self.log(f"[EXIT ORDERS] SL STOP-MARKET placed trigger @ {sl_trigger}")
        else:
            tick = self.tick if self.tick > 0 else 0.0
            offset = self.cfg.stop_limit_ticks * tick
            if side == "LONG":
                sl_limit = clamp_tick(max(sl_trigger - offset, 0.0), tick) if tick > 0 else sl_trigger
            else:
                sl_limit = clamp_tick(sl_trigger + offset, tick) if tick > 0 else sl_trigger

            sl = self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side=sl_side,
                type="STOP",                # stop-limit
                timeInForce="GTC",
                quantity=str(qty),
                price=f"{sl_limit:.10f}",
                stopPrice=f"{sl_trigger:.10f}",
                reduceOnly="true",
                workingType="MARK_PRICE",
                newOrderRespType="RESULT"
            )
            self.log(f"[EXIT ORDERS] SL STOP-LIMIT placed trigger @ {sl_trigger} limit @ {sl_limit}")

    def detect_bar_open(self) -> Optional[int]:
        now_ms = server_time_ms(self.client)
        open_ms = (now_ms // self.interval_ms) * self.interval_ms
        age_ms = now_ms - open_ms
        if self.last_seen_open_ms == open_ms:
            return None
        if age_ms <= int(self.cfg.entry_open_grace_sec * 1000):
            self.last_seen_open_ms = open_ms
            return open_ms
        return None

    def on_bar_open(self, bar_open_ms: int):
        # Fetch enough history to compute indicators on last closed bar
        lookback = max(self.cfg.rsi_period, self.cfg.atr_period) + 5
        df = get_klines(self.client, self.cfg.symbol, self.cfg.interval, limit=lookback)
        if len(df) < lookback - 2:
            self.log("[SKIP] insufficient klines")
            return

        # last closed bar is df.iloc[-2], forming is df.iloc[-1]
        last_closed = df.iloc[-2]
        forming = df.iloc[-1]
        # sanity: forming.open should equal last_closed.close
        ref_open = float(forming["open"])
        last_close = float(last_closed["close"])

        # Indicators on fully closed data
        df_sig = df.iloc[:-1].copy()
        rsi = calc_rsi(df_sig["close"], self.cfg.rsi_period)
        atr = calc_atr(df_sig["high"], df_sig["low"], df_sig["close"], self.cfg.atr_period)
        rsi_last = float(rsi.iloc[-1])
        atr_last = float(atr.iloc[-1])

        self.log(f"[BAR OPEN] {self.cfg.symbol} {self.cfg.interval} | rsi={rsi_last:.2f} hi={self.cfg.rsi_high} lo={self.cfg.rsi_low} | atr={atr_last:.8f} ref_open={ref_open:.8f}")

        if self.in_position:
            self.log("[SKIP] already in position")
            return

        # Signal from last closed bar
        side = None
        if rsi_last > self.cfg.rsi_high:
            side = "LONG"
        elif rsi_last < self.cfg.rsi_low:
            side = "SHORT"

        if side is None:
            self.log("[NO SIGNAL] RSI within band")
            return

        dist = self.cfg.atr_multiple * atr_last
        if not math.isfinite(dist) or dist <= 0:
            self.log("[SKIP] invalid ATR distance")
            return

        qty = self.compute_qty(ref_open)
        if qty <= 0:
            self.log("[SKIP] qty rounded to 0; adjust notional/quantity")
            return

        if side == "LONG":
            tp = clamp_tick(ref_open + dist, self.tick)
            sl = clamp_tick(ref_open - dist, self.tick)
            entry_side = "BUY"
        else:
            tp = clamp_tick(ref_open - dist, self.tick)
            sl = clamp_tick(ref_open + dist, self.tick)
            entry_side = "SELL"

        self.log(f"[ENTRY @ OPEN] {side} qty={qty} ref_open={ref_open:.8f} (prev_close={last_close:.8f}) TP={tp:.8f} SL={sl:.8f}")
        try:
            res = market_order(self.client, self.cfg.symbol, entry_side, qty, reduce_only=False)
            self.log(f"Entry: {res}")
        except BinanceAPIException as e:
            self.log(f"[ENTRY FAILED] {e}")
            return

        # Place exits immediately
        try:
            self.place_exit_orders(side, qty, tp, sl)
        except BinanceAPIException as e:
            self.log(f"[EXIT ORDERS FAILED] {e} | Fail-safe close at market")
            try:
                self.close_position_market()
            except Exception:
                pass
            return

        # Update state
        self.in_position = True
        self.side = side
        self.entry_open = ref_open
        self.tp = tp
        self.sl = sl
        self.bar_open_ms = bar_open_ms
        self.bar_close_ms = bar_open_ms + self.interval_ms
        self.log(f"[STATE] bar_open_ms={self.bar_open_ms} bar_close_ms={self.bar_close_ms}")

    def close_position_market(self):
        amt = position_amt(self.client, self.cfg.symbol)
        if abs(amt) < 1e-12:
            return None
        side = "SELL" if amt > 0 else "BUY"
        qty = round_step(abs(amt), self.step)
        if qty <= 0:
            return None
        return market_order(self.client, self.cfg.symbol, side, qty, reduce_only=True)

    def manage_position(self):
        if not self.in_position:
            return
        # Check if TP/SL already flattened position
        amt = position_amt(self.client, self.cfg.symbol)
        if abs(amt) < 1e-12:
            self.log("[EXIT] Position flat via TP/SL; cancelling residuals.")
            cancel_all_open_orders(self.client, self.cfg.symbol)
            self._reset_pos()
            return

        # Time-stop at bar close
        now = server_time_ms(self.client)
        if now >= self.bar_close_ms:
            self.log("[EXIT] Bar close reached, cancel exits and close at market.")
            cancel_all_open_orders(self.client, self.cfg.symbol)
            res = self.close_position_market()
            self.log(f"Close @bar_close: {res}")
            self._reset_pos()

    def _reset_pos(self):
        self.in_position = False
        self.side = None
        self.entry_open = None
        self.tp = None
        self.sl = None
        self.bar_open_ms = None
        self.bar_close_ms = None

    def run(self):
        self.log(f"Bot running | {self.cfg.symbol} {self.cfg.interval} | RSI({self.cfg.rsi_period}) > {self.cfg.rsi_high} / < {self.cfg.rsi_low} | ATR({self.cfg.atr_period}) x {self.cfg.atr_multiple}")
        while True:
            try:
                open_ms = self.detect_bar_open()
                if open_ms is not None:
                    self.on_bar_open(open_ms)
                self.manage_position()
                time.sleep(self.cfg.poll_interval_sec)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:
                print("Main loop error:", e)
                traceback.print_exc()
                time.sleep(2.0)


def main():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Please set BINANCE_API_KEY and BINANCE_API_SECRET.")

    cfg = Config(
        symbol=os.getenv("BOT_SYMBOL", "SPXUSDT"),
        interval=os.getenv("BOT_INTERVAL", "5m"),
        rsi_period=int(os.getenv("BOT_RSI_PERIOD", "30")),
        rsi_high=float(os.getenv("BOT_RSI_HIGH", "60")),
        rsi_low=float(os.getenv("BOT_RSI_LOW", "40")),
        atr_period=int(os.getenv("BOT_ATR_PERIOD", "30")),
        atr_multiple=float(os.getenv("BOT_ATR_MULTIPLE", "0.95")),
        use_fixed_notional=(os.getenv("BOT_USE_FIXED_NOTIONAL", "true").lower() == "true"),
        fixed_usdt_notional=float(os.getenv("BOT_FIXED_USDT_NOTIONAL", "50")),
        fixed_quantity=float(os.getenv("BOT_FIXED_QUANTITY", "1")),
        leverage=int(os.getenv("BOT_LEVERAGE", "5")),
        poll_interval_sec=float(os.getenv("BOT_POLL_INTERVAL_SEC", "0.5")),
        entry_open_grace_sec=float(os.getenv("BOT_ENTRY_OPEN_GRACE_SEC", "10.0")),
        sl_use_stop_market=(os.getenv("BOT_SL_USE_STOP_MARKET", "false").lower() == "true"),
        stop_limit_ticks=int(os.getenv("BOT_STOP_LIMIT_TICKS", "3")),
        verbose=(os.getenv("BOT_VERBOSE", "true").lower() == "true"),
    )

    client = Client(api_key, api_secret)
    bot = RSIAtrBot(client, cfg)
    bot.run()


if __name__ == "__main__":
    main()
