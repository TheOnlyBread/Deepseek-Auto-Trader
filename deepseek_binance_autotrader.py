# deepseek_binance_autotrader.py
# Render-ready FastAPI panel + Binance USDⓈ-M Futures *testnet* bot (no DB).
# Updates:
#   - DeepSeek HTTP/401/429/etc => safe "flat" signal + /api/last_error message (no crash).
#   - Unwrap tenacity.RetryError to show root cause.
#   - Keeps previous fixes: proper signature, time sync, pandas list indexer, no DB, panel.

import os
import time
import json
import math
from threading import Thread, Lock
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError as TenacityRetryError

# -------------------- ENV --------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
KLIMIT = int(os.getenv("KLIMIT", "250"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
NOTIONAL_PER_TRADE_USDT = float(os.getenv("NOTIONAL_PER_TRADE_USDT", "50"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.62"))
DECISION_COOLDOWN_SEC = int(os.getenv("DECISION_COOLDOWN_SEC", "20"))
MAX_OPEN_SECONDS = int(os.getenv("MAX_OPEN_SECONDS", "3600"))
RISK_WORKING_TYPE = "MARK_PRICE"
REGIME_LOOKBACK_DAYS = int(os.getenv("REGIME_LOOKBACK_DAYS", "365"))
REGIME_INTERVAL = os.getenv("REGIME_INTERVAL", "1h")
REGIME_REFRESH_MINUTES = int(os.getenv("REGIME_REFRESH_MINUTES", "10"))
USE_HYBRID_GATES = os.getenv("USE_HYBRID_GATES", "true").lower() == "true"
SIZE_SCALE_MIN = float(os.getenv("SIZE_SCALE_MIN", "0.35"))
SIZE_SCALE_MAX = float(os.getenv("SIZE_SCALE_MAX", "1.25"))
ANALYTICS_HOST = os.getenv("ANALYTICS_HOST", "0.0.0.0")
ANALYTICS_PORT = int(os.getenv("ANALYTICS_PORT", os.getenv("PORT", "10000")))
RUN_TRADER = os.getenv("RUN_TRADER", "true").lower() == "true"
EQUITY_USDT = float(os.getenv("EQUITY_USDT", "1000"))
ROI_TRADES_PER_DAY = float(os.getenv("ROI_TRADES_PER_DAY", "20"))
ROI_WIN_RATE = float(os.getenv("ROI_WIN_RATE", "0.54"))
ROI_AVG_WIN_R = float(os.getenv("ROI_AVG_WIN_R", "0.9"))
ROI_AVG_LOSS_R = float(os.getenv("ROI_AVG_LOSS_R", "0.8"))
ROI_RISK_PER_TRADE_USDT = float(os.getenv("ROI_RISK_PER_TRADE_USDT", "5"))

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/chat/completions"
BINANCE_FAPI_BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"

# -------------------- In-memory state --------------------
DECISIONS = deque(maxlen=5000)
REGIME_SNAPSHOT: Dict[str, Any] = {}
ACCOUNT = {"balance_usdt": None, "pnl24h": None}
SERVICE_NOTICE = ""
LAST_ERROR = ""
LOCK = Lock()

def set_notice(msg: str):
    global SERVICE_NOTICE
    with LOCK:
        SERVICE_NOTICE = msg

def get_notice() -> str:
    with LOCK:
        return SERVICE_NOTICE

def set_last_error(msg: str):
    global LAST_ERROR
    with LOCK:
        LAST_ERROR = msg[:2000]  # cap

def get_last_error() -> str:
    with LOCK:
        return LAST_ERROR

def put_decision(row: dict):
    with LOCK:
        DECISIONS.append(row)

def latest_decisions(n: int) -> List[dict]:
    with LOCK:
        return list(list(DECISIONS)[-n:])[::-1]

def put_regime(snap: dict):
    with LOCK:
        REGIME_SNAPSHOT.clear()
        REGIME_SNAPSHOT.update(snap)

def get_regime() -> dict:
    with LOCK:
        return dict(REGIME_SNAPSHOT)

def put_account(balance: Optional[float], pnl24h: Optional[float]):
    with LOCK:
        ACCOUNT["balance_usdt"] = balance
        ACCOUNT["pnl24h"] = pnl24h

def get_account() -> dict:
    with LOCK:
        return dict(ACCOUNT)

# -------------------- Helpers --------------------
class BinanceHTTPError(Exception):
    def __init__(self, status: int, body: str, where: str):
        super().__init__(f"{where} | HTTP {status} | {body}")
        self.status = status
        self.body = body
        self.where = where

class LogicSkip(Exception): pass

def _ts_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def _hmac_sha256(secret: str, msg: str) -> str:
    import hashlib, hmac
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

def as_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def interval_to_ms(interval: str) -> int:
    unit = interval[-1]; n = int(interval[:-1])
    if unit == 'm': return n * 60 * 1000
    if unit == 'h': return n * 60 * 60 * 1000
    if unit == 'd': return n * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")

def pct_rank(x: np.ndarray, v: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0: return 0.5
    return float((x <= v).sum()) / float(len(x))

# -------------------- Binance client --------------------
class BinanceFutures:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.ak = api_key
        self.sk = api_secret
        self.base = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.ak})
        self.time_offset_ms = 0
        try:
            self._sync_time()
        except Exception as e:
            print(f"[TimeSync] warn: {e}")

    def _sync_time(self):
        url = self.base + "/fapi/v1/time"
        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        server_time = int(r.json()["serverTime"])
        local = _ts_ms()
        self.time_offset_ms = server_time - local
        print(f"[TimeSync] offset_ms={self.time_offset_ms}")

    def _signed_qs(self, params: Dict[str, Any]) -> str:
        base_params = dict(params or {})
        if abs(self.time_offset_ms) > 60_000:
            try: self._sync_time()
            except Exception as e: print(f"[TimeSync] retry failed: {e}")
        base_params["timestamp"] = _ts_ms() + self.time_offset_ms
        base_params.setdefault("recvWindow", 5000)
        items = sorted(base_params.items(), key=lambda kv: kv[0])
        qs = urlencode(items, doseq=True)
        sig = _hmac_sha256(self.sk, qs)
        return f"{qs}&signature={sig}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.25, min=0.25, max=1.8),
           retry=retry_if_exception_type(BinanceHTTPError))
    def _request(self, method: str, path: str, params: Dict[str, Any] = None, signed: bool = False):
        url = self.base + path
        where = f"{method} {path}"
        try:
            if signed:
                final_qs = self._signed_qs(params or {})
                if method in ("GET","DELETE"):
                    full = f"{url}?{final_qs}"
                    resp = self.session.request(method, full, timeout=20)
                elif method == "POST":
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    resp = self.session.post(url, data=final_qs, headers=headers, timeout=20)
                else:
                    raise ValueError("Unsupported method")
            else:
                if method in ("GET","DELETE"):
                    resp = self.session.request(method, url, params=params, timeout=20)
                elif method == "POST":
                    resp = self.session.post(url, params=params, timeout=20)
                else:
                    raise ValueError("Unsupported method")
        except requests.RequestException as e:
            set_last_error(f"{where} | RequestException: {e}")
            raise BinanceHTTPError(599, f"network error: {e}", where)

        if resp.status_code >= 400:
            body = resp.text[:500]
            set_last_error(f"{where} | HTTP {resp.status_code} | {body}")
            if resp.status_code == 400 and ("-1021" in body or "Timestamp" in body):
                try: self._sync_time()
                except Exception: pass
            raise BinanceHTTPError(resp.status_code, body, where)
        return resp.json()

    # public
    def klines(self, symbol: str, interval: str, limit: int = 200):
        return self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int, step_limit: int = 1500) -> List[list]:
        out: List[list] = []
        itv = interval_to_ms(interval)
        cur = start_ms
        while cur < end_ms:
            params = {"symbol": symbol, "interval": interval, "limit": step_limit, "startTime": cur,
                      "endTime": min(cur + itv * step_limit, end_ms)}
            chunk = self._request("GET", "/fapi/v1/klines", params)
            if not chunk:
                break
            out.extend(chunk)
            last_close = int(chunk[-1][6]); nxt = last_close + 1
            if nxt <= cur: break
            cur = nxt
            time.sleep(0.05)
        return out

    def exchange_info(self, symbol: Optional[str] = None):
        data = self._request("GET", "/fapi/v1/exchangeInfo")
        if symbol:
            for s in data.get("symbols", []):
                if s.get("symbol") == symbol:
                    return s
            return None
        return data

    # signed
    def change_leverage(self, symbol: str, leverage: int):
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def new_order(self, **params):
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all(self, symbol: str):
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    def position_risk(self):
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)

    def income_history(self, startTime: Optional[int] = None, endTime: Optional[int] = None, incomeType: Optional[str] = None):
        p = {}
        if startTime: p["startTime"] = startTime
        if endTime: p["endTime"] = endTime
        if incomeType: p["incomeType"] = incomeType
        return self._request("GET", "/fapi/v1/income", p, signed=True)

    def balances(self):
        return self._request("GET", "/fapi/v2/balance", signed=True)

# -------------------- Indicators / quantization --------------------
@dataclass
class SymbolFilters:
    step_size: float
    min_qty: float
    tick_size: float

def parse_filters(ex_info: Dict[str, Any]) -> SymbolFilters:
    step_size = min_qty = tick_size = 0.0
    for f in ex_info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            step_size = float(f["stepSize"]); min_qty = float(f["minQty"])
        elif f.get("filterType") == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
    if not (step_size and tick_size):
        raise ValueError("Missing LOT_SIZE or PRICE_FILTER in exchangeInfo")
    return SymbolFilters(step_size=step_size, min_qty=min_qty, tick_size=tick_size)

def quantize_qty(qty: float, step: float, min_qty: float) -> float:
    if qty < min_qty: return 0.0
    steps = math.floor(qty / step)
    return round(steps * step, 12)

def quantize_price(px: float, tick: float) -> float:
    ticks = math.floor(px / tick)
    return round(ticks * tick, 12)

# -------------------- DeepSeek (safe JSON; tolerant to HTTP errors) --------------------
class DeepSeek:
    def __init__(self, api_key: str):
        self.ak = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.ak}", "Content-Type": "application/json"})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=1.5))
    def decide(self, features: Dict[str, float]) -> Dict[str, Any]:
        sys = ("You are a trading signal engine. Output STRICT JSON with keys: "
               "signal ('long','short','flat'), confidence (0..1 float), sl_pct (positive), tp_pct (positive).")
        usr = "JSON ONLY:\n" + json.dumps(features, separators=(",", ":"))
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            "temperature": 0.15, "max_tokens": 150, "response_format": {"type": "json_object"},
        }
        try:
            r = self.session.post(DEEPSEEK_BASE + DEEPSEEK_CHAT_PATH, data=json.dumps(payload), timeout=20)
            if r.status_code >= 400:
                # Write exact status + trimmed body; return flat to avoid crashing loop.
                body = r.text[:300]
                set_last_error(f"DeepSeek decide HTTP {r.status_code} | {body}")
                # Return a safe no-trade signal
                return {"signal": "flat", "confidence": 0.0, "sl_pct": 0.01, "tp_pct": 0.02}
            content = r.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (requests.RequestException, ValueError, KeyError) as e:
            set_last_error(f"DeepSeek decide exception: {e}")
            return {"signal": "flat", "confidence": 0.0, "sl_pct": 0.01, "tp_pct": 0.02}

        sig = parsed.get("signal", "flat")
        conf = float(parsed.get("confidence", 0.0))
        slp = float(parsed.get("sl_pct", 0.01))
        tpp = float(parsed.get("tp_pct", 0.02))
        if sig not in ("long", "short", "flat"): sig = "flat"
        return {"signal": sig, "confidence": conf, "sl_pct": abs(slp), "tp_pct": abs(tpp)}

# -------------------- Feature builders --------------------
def build_features(klines: list):
    cols12 = ["ot","o","h","l","c","v","ct","qv","n","tbv","tqv","x"]
    df = pd.DataFrame(klines, columns=cols12)
    for col in ("o","h","l","c","v"): df[col] = df[col].astype(float)
    df["ret1"] = df["c"].pct_change().fillna(0)
    df["hl_range"] = (df["h"] - df["l"]) / df["o"].replace(0, np.nan)
    df["ema_fast"] = df["c"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=21, adjust=False).mean()
    df["rsi14"] = rsi(df["c"], 14)
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["c"]
    df["vol_z"] = (df["v"] - df["v"].rolling(30).mean()) / (df["v"].rolling(30).std() + 1e-9)
    latest = df.iloc[-1]
    feat = {"ret1": float(latest["ret1"]), "hl_range": float(latest["hl_range"]),
            "ema_diff": float(latest["ema_diff"]), "rsi14": float(latest["rsi14"]),
            "vol_z": float(latest["vol_z"]), "price": float(latest["c"])}
    for k,v in list(feat.items()):
        if not np.isfinite(v): feat[k] = 0.0
    return df, feat

# -------------------- Regime analyzer --------------------
@dataclass
class RegimeSnapshot:
    ts: float
    features: Dict[str, float]
    info: Dict[str, Any]

class RegimeAnalyzer:
    def __init__(self, client: BinanceFutures):
        self.b = client
        self.snapshot: Optional[RegimeSnapshot] = None
        self.lock = Lock()
        self.run = True
        Thread(target=self._loop, daemon=True).start()

    def stop(self): self.run = False

    def get_snapshot(self) -> Optional[RegimeSnapshot]:
        with self.lock:
            return self.snapshot

    def _fetch_year(self, symbol: str, interval: str) -> pd.DataFrame:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=REGIME_LOOKBACK_DAYS)
        kl = self.b.klines_range(symbol, interval, int(start.timestamp()*1000), int(end.timestamp()*1000), 1500)
        if not kl or len(kl) < 100:
            raise LogicSkip(f"Not enough history for {symbol}")
        cols12 = ["ot","o","h","l","c","v","ct","qv","n","tbv","tqv","x"]
        df = pd.DataFrame(kl, columns=cols12)
        for c in ("o","h","l","c","v"): df[c] = df[c].astype(float)
        df["dt"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
        df.set_index("dt", inplace=True)
        # LIST (not set) for pandas indexer:
        return df[["o","h","l","c","v","ct"]]

    def _compute_once(self) -> RegimeSnapshot:
        btc = self._fetch_year("BTCUSDT", REGIME_INTERVAL)
        alt = self._fetch_year(ACTIVE_SYMBOL, REGIME_INTERVAL)
        btc = btc.sort_index(); alt = alt.sort_index()
        joined = btc[["c"]].rename(columns={"c":"btc_c"}).join(
            alt[["c"]].rename(columns={"c":"alt_c"}), how="inner").dropna()
        joined["btc_r"] = joined["btc_c"].pct_change()
        joined["alt_r"] = joined["alt_c"].pct_change()

        btc_close = joined["btc_c"]
        btc_ema50 = btc_close.ewm(span=50, adjust=False).mean()
        btc_ema200 = btc_close.ewm(span=200, adjust=False).mean()
        btc_trend_raw = (btc_ema50 - btc_ema200) / btc_close
        btc_trend_state = np.sign(btc_trend_raw.iloc[-1])

        win = int(24*30)
        btc_vol = joined["btc_r"].rolling(win).std()
        vol_latest = float(btc_vol.iloc[-1]); vol_hist = btc_vol.values
        vol_pct = pct_rank(vol_hist[np.isfinite(vol_hist)], vol_latest)

        w7 = 24*7
        sub = joined.iloc[-w7:].dropna()
        if len(sub) < 10 or sub["btc_r"].var() == 0:
            corr7 = beta7 = 0.0
        else:
            corr7 = float(sub["alt_r"].corr(sub["btc_r"]))
            beta7 = float(np.cov(sub["alt_r"], sub["btc_r"])[0,1] / (sub["btc_r"].var() + 1e-12))

        alt_cum = (1 + sub["alt_r"]).prod() - 1 if len(sub) else 0.0
        btc_cum = (1 + sub["btc_r"]).prod() - 1 if len(sub) else 0.0
        rel_str = float(alt_cum - beta7 * btc_cum)

        tmp = pd.DataFrame({"c": joined["btc_c"]})
        tmp["rsi14"] = rsi(tmp["c"], 14); btc_rsi = float(tmp["rsi14"].iloc[-1])

        trend_mag = abs(float(btc_trend_raw.iloc[-1]))
        is_chop = 1.0 if (trend_mag < 0.0015 and vol_pct > 0.55) else 0.0

        feat = {
            "btc_trend_state": float(btc_trend_state),
            "btc_trend_mag": float(trend_mag),
            "btc_vol_pct": float(vol_pct),
            "btc_rsi14": float(btc_rsi),
            "corr7": float(corr7),
            "beta7": float(beta7),
            "alt_rel_strength_7d": float(rel_str),
            "regime_is_chop": float(is_chop),
        }
        info = {"btc_price": float(btc_close.iloc[-1]),
                "alt_price": float(joined['alt_c'].iloc[-1]),
                "samples_1h": int(len(joined)), "window_corr_hours": int(w7)}
        return RegimeSnapshot(ts=time.time(), features=feat, info=info)

    def _loop(self):
        while self.run:
            try:
                snap = self._compute_once()
                with self.lock:
                    self.snapshot = snap
                put_regime({"ts": snap.ts, "features": snap.features, "info": snap.info})
                print(f"[Regime] refreshed @ {datetime.utcnow().isoformat(timespec='seconds')}Z")
            except Exception as e:
                set_last_error(f"Regime error: {e}")
                print(f"[Regime] error: {e}")
            time.sleep(REGIME_REFRESH_MINUTES * 60)

# -------------------- Trading --------------------
def quantize_qty_for_notional(price: float, notional: float, step: float, min_qty: float) -> float:
    raw = notional / price
    return quantize_qty(raw, step, min_qty)

@dataclass
class SymbolFiltersDat:
    step_size: float
    min_qty: float
    tick_size: float

class AutoTrader:
    def __init__(self, b: BinanceFutures, symbol: str):
        self.b = b
        self.symbol = symbol
        ex = self.b.exchange_info(self.symbol)
        if not ex:
            raise LogicSkip(f"Symbol {self.symbol} not found on this environment")
        self.filters = parse_filters(ex)
        self.position_open_time = None
        self.active_entry_price = None
        self.active_side = None
        self.active_qty = None
        self.regime = RegimeAnalyzer(self.b)
        try:
            self.b.change_leverage(self.symbol, LEVERAGE)
        except Exception as e:
            print(f"[Leverage] Warning: {e}")

    def _flat_position_amt(self) -> float:
        pos = self.b.position_risk()
        for p in pos:
            if p.get("symbol") == self.symbol:
                return as_float(p.get("positionAmt", "0"), 0.0)
        return 0.0

    def _cleanup_if_flat(self):
        amt = self._flat_position_amt()
        if abs(amt) < 1e-12:
            try: self.b.cancel_all(self.symbol)
            except Exception as e: print(f"[Cleanup] cancel_all: {e}")

    def _dynamic_notional(self, base_notional: float, regime_feat: Dict[str, float]) -> float:
        vol_pct = float(regime_feat.get("btc_vol_pct", 0.5))
        scale = SIZE_SCALE_MAX - (SIZE_SCALE_MAX - SIZE_SCALE_MIN) * vol_pct
        return max(5.0, base_notional * float(np.clip(scale, SIZE_SCALE_MIN, SIZE_SCALE_MAX)))

    def _hybrid_gate(self, desired_signal: str, regime_feat: Dict[str, float]) -> str:
        if not USE_HYBRID_GATES: return desired_signal
        trend = float(regime_feat.get("btc_trend_state", 0.0))
        volp  = float(regime_feat.get("btc_vol_pct", 0.5))
        corr  = float(regime_feat.get("corr7", 0.0))
        chop  = float(regime_feat.get("regime_is_chop", 0.0))
        if chop >= 0.5 and (volp >= 0.55 or abs(corr) <= 0.15): return "flat"
        if trend < 0 and volp >= 0.6 and desired_signal == "long": return "flat"
        if trend > 0 and corr >= 0.2 and desired_signal == "short": return "flat"
        return desired_signal

    def _place_entry_and_exits(self, signal: str, price: float, sl_pct: float, tp_pct: float, notional: float):
        qty = quantize_qty_for_notional(price, notional, self.filters.step_size, self.filters.min_qty)
        if qty <= 0: raise LogicSkip("Qty below LOT_SIZE; increase NOTIONAL_PER_TRADE_USDT")
        side = "BUY" if signal == "long" else "SELL"
        order = self.b.new_order(symbol=self.symbol, side=side, type="MARKET", quantity=f"{qty}")
        print(f"[Entry] {side} {qty} @ market → id={order.get('orderId')}")
        self.active_entry_price = price; self.active_side = side; self.active_qty = qty
        self.position_open_time = time.time()

        if signal == "long":
            sl_price = quantize_price(price * (1 - sl_pct), self.filters.tick_size)
            tp_price = quantize_price(price * (1 + tp_pct), self.filters.tick_size)
            sl_side = tp_side = "SELL"
        else:
            sl_price = quantize_price(price * (1 + sl_pct), self.filters.tick_size)
            tp_price = quantize_price(price * (1 - tp_pct), self.filters.tick_size)
            sl_side = tp_side = "BUY"

        self.b.new_order(symbol=self.symbol, side=sl_side, type="STOP_MARKET",
                         stopPrice=f"{sl_price}", closePosition="true", workingType=RISK_WORKING_TYPE)
        self.b.new_order(symbol=self.symbol, side=tp_side, type="TAKE_PROFIT_MARKET",
                         stopPrice=f"{tp_price}", closePosition="true", workingType=RISK_WORKING_TYPE)
        print(f"[Exits] SL {sl_side} @{sl_price} | TP {tp_side} @{tp_price}")

    def _close_if_timeout(self, current_price: float):
        if self.position_open_time and time.time() - self.position_open_time > MAX_OPEN_SECONDS:
            amt = self._flat_position_amt()
            if abs(amt) > 0:
                side = "SELL" if amt > 0 else "BUY"
                self.b.cancel_all(self.symbol)
                self.b.new_order(symbol=self.symbol, side=side, type="MARKET", quantity=f"{abs(amt)}")
                print("[Panic] Closed due to MAX_OPEN_SECONDS")
            self.active_entry_price = None; self.active_side = None; self.active_qty = None
            self.position_open_time = None

    def step(self):
        kl = self.b.klines(self.symbol, INTERVAL, limit=KLIMIT)
        if not kl or len(kl) < 50: raise LogicSkip("Not enough bars")
        _, feat = build_features(kl); price = feat["price"]

        self._close_if_timeout(price)
        if abs(self._flat_position_amt()) > 0: raise LogicSkip("Position active; waiting")

        snap = self.regime.get_snapshot()
        if not snap: raise LogicSkip("Regime snapshot not ready")

        all_feat = {**feat}
        for k, v in snap.features.items(): all_feat[f"X_{k}"] = float(v)

        decision = DeepSeek(DEEPSEEK_API_KEY).decide(all_feat)
        sig, conf = decision["signal"], decision["confidence"]
        slp = max(0.001, decision["sl_pct"]); tpp = max(0.001, decision["tp_pct"])
        gated_sig = self._hybrid_gate(sig, snap.features)

        put_decision({"ts": time.time(), "price": price, "llm_signal": sig,
                      "confidence": conf, "gated_signal": gated_sig,
                      "sl_pct": slp, "tp_pct": tpp})

        print(f"[Decision] LLM={sig}({conf:.3f}) → Gate={gated_sig} | sl={slp:.3f} tp={tpp:.3f} | px={price}")

        if gated_sig == "flat" or conf < CONFIDENCE_THRESHOLD:
            raise LogicSkip("No trade (confidence low / gate flat)")

        dyn_notional = self._dynamic_notional(NOTIONAL_PER_TRADE_USDT, snap.features)
        self._place_entry_and_exits(gated_sig, price, slp, tpp, dyn_notional)
        time.sleep(1); self._cleanup_if_flat()

    def loop(self):
        print(f"[Bot] Env={'TESTNET' if USE_TESTNET else 'MAINNET'} | Symbol={self.symbol} | Lev={LEVERAGE} | Notional={NOTIONAL_PER_TRADE_USDT}")
        while True:
            try:
                self.step()
            except LogicSkip as e:
                print(f"[Skip] {e}")
            except BinanceHTTPError as e:
                set_last_error(str(e))
                print(f"[Binance] {e}")
            except TenacityRetryError as e:
                root = repr(getattr(e.last_attempt, "exception", lambda: None)())
                set_last_error(f"RetryError root cause: {root}")
                print(f"[RetryError] {root}")
            except requests.RequestException as e:
                set_last_error(f"Requests error: {e}")
                print(f"[HTTP] {e}")
            except Exception as e:
                set_last_error(f"Unhandled: {e}")
                print(f"[Error] {type(e).__name__}: {e}")
            time.sleep(DECISION_COOLDOWN_SEC)

# -------------------- Account cache writer --------------------
class AccountCacheWriter:
    def __init__(self, bclient: BinanceFutures, interval_sec: int = 300):
        self.b = bclient; self.interval = max(60, interval_sec)
        Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            bal = pnl = None
            try:
                res = self.b.balances()
                usdt = [x for x in res if x.get("asset") == "USDT"]
                if usdt: bal = float(usdt[0].get("balance", 0.0))
            except Exception as e:
                set_last_error(f"balances: {e}")
            try:
                since = int((datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000)
                inc = self.b.income_history(startTime=since, incomeType="REALIZED_PNL")
                pnl = sum(float(x.get("income", 0.0)) for x in inc)
            except Exception as e:
                set_last_error(f"pnl24h: {e}")
            put_account(bal, pnl)
            time.sleep(self.interval)

# -------------------- FastAPI Panel --------------------
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

def build_app():
    app = FastAPI()

    DASH = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>AutoTrader Panel</title><meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;color:#111;background:#fafafa;margin:0;padding:20px}
.card{background:#fff;border-radius:16px;box-shadow:0 1px 8px rgba(0,0,0,.06);padding:16px;margin-bottom:16px}
h1{margin:0 0 12px 0} table{width:100%;border-collapse:collapse} th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}
.badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;background:#eef}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.notice{color:#555;margin:8px 0}
.err{color:#b00;white-space:pre-wrap}
</style></head><body>
<h1>DeepSeek × Binance AutoTrader</h1>
<div class="notice" id="notice"></div>
<div class="grid">
  <div class="card"><h3>Status</h3><div id="status"></div></div>
  <div class="card"><h3>ROI Model (assumptions)</h3><div id="roi"></div></div>
  <div class="card"><h3>Resource Estimate</h3><div id="res"></div></div>
</div>
<div class="card">
  <h3>Latest Decisions</h3>
  <table id="dec"><thead><tr><th>Time</th><th>Price</th><th>LLM</th><th>Conf</th><th>Gate</th></tr></thead><tbody></tbody></table>
</div>
<div class="card">
  <h3>Regime Snapshot</h3>
  <pre id="regime" style="white-space:pre-wrap"></pre>
</div>
<div class="card">
  <h3>Last Error</h3>
  <pre class="err" id="lerr"></pre>
</div>
<script>
async function load(){
  const s = await fetch('/api/status').then(r=>r.json());
  const r = await fetch('/api/roi').then(r=>r.json());
  const re= await fetch('/api/resources').then(r=>r.json());
  const d = await fetch('/api/decisions?limit=20').then(r=>r.json());
  const g = await fetch('/api/regime').then(r=>r.json());
  const le= await fetch('/api/last_error').then(r=>r.text());
  const n = await fetch('/api/notice').then(r=>r.text());
  document.getElementById('notice').textContent = n;
  document.getElementById('status').innerHTML =
    `<div><span class="badge">Symbol</span> ${s.symbol}</div>
     <div><span class="badge">Env</span> ${s.env}</div>
     <div><span class="badge">Leverage</span> ${s.leverage}</div>
     <div><span class="badge">Notional</span> ${s.notional} USDT</div>
     <div><span class="badge">Balance (USDT)</span> ${s.balance_usdt ?? '—'}</div>
     <div><span class="badge">24h Realized PnL</span> ${s.pnl24h ?? '—'}</div>`;
  document.getElementById('roi').innerHTML =
    `<div><b>Exp. Daily PnL</b>: ${r.expected_daily_pnl_usdt.toFixed(2)} USDT</div>
     <div>Trades/Day: ${r.trades_per_day}, Win%: ${(r.win_rate*100).toFixed(0)}%, AvgWinR: ${r.avg_win_r}, AvgLossR: ${r.avg_loss_r}, Risk/Trade: ${r.risk_per_trade}</div>
     <div><b>Exp. Daily ROI</b>: ${(r.expected_daily_roi*100).toFixed(2)}%</div>`;
  document.getElementById('res').innerHTML =
    `<div>Yearly bars (1h): ${re.year_bars}</div>
     <div>Memory (data): ~${re.mem_mb.toFixed(2)} MB</div>
     <div>CPU Needs: ${re.cpu_needs}</div>`;
  const tb = document.querySelector('#dec tbody'); tb.innerHTML='';
  d.rows.forEach(x=>{
    tb.insertAdjacentHTML('beforeend', `<tr>
      <td>${new Date(x.ts*1000).toLocaleTimeString()}</td>
      <td>${x.price}</td>
      <td>${x.llm_signal}</td>
      <td>${x.confidence.toFixed(3)}</td>
      <td>${x.gated_signal}</td>
    </tr>`);
  });
  document.getElementById('regime').textContent = JSON.stringify(g, null, 2);
  document.getElementById('lerr').textContent = le;
}
setInterval(load, 4000); load();
</script>
</body></html>
    """

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home(): return HTMLResponse(DASH)

    @app.head("/")
    def head_home(): return Response(status_code=200)

    @app.get("/healthz")
    def healthz(): return {"ok": True}

    @app.get("/api/decisions")
    def api_decisions(limit: int = 50):
        return {"rows": latest_decisions(limit)}

    @app.get("/api/regime")
    def api_regime():
        return get_regime() or {}

    @app.get("/api/status")
    def api_status():
        acc = get_account()
        return {"symbol": ACTIVE_SYMBOL, "env": "TESTNET" if USE_TESTNET else "MAINNET",
                "leverage": LEVERAGE, "notional": NOTIONAL_PER_TRADE_USDT, **acc}

    @app.get("/api/roi")
    def api_roi():
        ev = ROI_WIN_RATE*(ROI_AVG_WIN_R*ROI_RISK_PER_TRADE_USDT) - (1-ROI_WIN_RATE)*(ROI_AVG_LOSS_R*ROI_RISK_PER_TRADE_USDT)
        pnl = ev*ROI_TRADES_PER_DAY; roi = pnl/max(1.0, EQUITY_USDT)
        return {"expected_daily_pnl_usdt": pnl, "expected_daily_roi": roi,
                "trades_per_day": ROI_TRADES_PER_DAY, "win_rate": ROI_WIN_RATE,
                "avg_win_r": ROI_AVG_WIN_R, "avg_loss_r": ROI_AVG_LOSS_R,
                "risk_per_trade": ROI_RISK_PER_TRADE_USDT}

    @app.get("/api/resources")
    def api_resources():
        year_bars = 8760; cols=10
        mem_mb = (year_bars*cols*8*2)/(1024**2)
        return {"year_bars": year_bars, "mem_mb": mem_mb,
                "cpu_needs": "1 vCPU / 512MB is fine; 2 vCPU / 1–2GB for more symbols or 10–15s loop."}

    @app.get("/api/last_error")
    def api_last_error():
        return get_last_error()

    @app.get("/api/notice")
    def api_notice():
        return get_notice()

    return app

# -------------------- Startup wiring --------------------
ACTIVE_SYMBOL = SYMBOL

def start_everything():
    global ACTIVE_SYMBOL
    if not RUN_TRADER:
        set_notice("RUN_TRADER=false (panel only).")
        return

    if not (DEEPSEEK_API_KEY and BINANCE_API_KEY and BINANCE_API_SECRET):
        set_notice("Missing keys (DEEPSEEK_API_KEY, BINANCE_API_KEY, BINANCE_API_SECRET). Trader disabled.")
        print("Missing keys; trader not started.")
        return

    b = BinanceFutures(BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_FAPI_BASE)

    try:
        ex = b.exchange_info(SYMBOL)
        if not ex:
            fallback = "BTCUSDT"
            set_notice(f"Symbol '{SYMBOL}' not available on {'TESTNET' if USE_TESTNET else 'MAINNET'}. "
                       f"Falling back to '{fallback}'.")
            ACTIVE_SYMBOL = fallback
        else:
            ACTIVE_SYMBOL = SYMBOL
    except BinanceHTTPError as e:
        set_last_error(f"exchange_info error: {e}")
        ACTIVE_SYMBOL = "BTCUSDT"
        set_notice(f"exchange_info failed for {SYMBOL}; falling back to BTCUSDT.")

    AccountCacheWriter(b, interval_sec=300)
    trader = AutoTrader(b, ACTIVE_SYMBOL)
    Thread(target=trader.loop, daemon=True).start()

start_everything()
app = build_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ANALYTICS_HOST, port=ANALYTICS_PORT, log_level="info")
