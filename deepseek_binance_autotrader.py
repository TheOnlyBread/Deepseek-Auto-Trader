# deepseek_binance_autotrader.py
# Render-ready: Worker runs this file; Web runs panel.py (panel:app)
# Deps: requests, python-dotenv, tenacity, pandas, numpy, fastapi, uvicorn, psycopg[binary]

import os
import time
import hmac
import hashlib
import json
import math
from threading import Thread, Lock
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, Optional, List

import requests
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# --------------------------
# Config & ENV
# --------------------------
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
SYMBOL = os.getenv("SYMBOL", "1000PEPEUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
KLIMIT = int(os.getenv("KLIMIT", "250"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
NOTIONAL_PER_TRADE_USDT = float(os.getenv("NOTIONAL_PER_TRADE_USDT", "50"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.62"))
DECISION_COOLDOWN_SEC = int(os.getenv("DECISION_COOLDOWN_SEC", "20"))
MAX_OPEN_SECONDS = int(os.getenv("MAX_OPEN_SECONDS", "3600"))
RISK_WORKING_TYPE = "MARK_PRICE"
DAILY_MAX_LOSS_USDT = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))

# Yearly regime
REGIME_LOOKBACK_DAYS = int(os.getenv("REGIME_LOOKBACK_DAYS", "365"))
REGIME_INTERVAL = os.getenv("REGIME_INTERVAL", "1h")
REGIME_REFRESH_MINUTES = int(os.getenv("REGIME_REFRESH_MINUTES", "10"))
USE_HYBRID_GATES = os.getenv("USE_HYBRID_GATES", "true").lower() == "true"
SIZE_SCALE_MIN = float(os.getenv("SIZE_SCALE_MIN", "0.35"))
SIZE_SCALE_MAX = float(os.getenv("SIZE_SCALE_MAX", "1.25"))

# Analytics panel
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "false").lower() == "true"
ANALYTICS_HOST = os.getenv("ANALYTICS_HOST", "0.0.0.0")
ANALYTICS_PORT = int(os.getenv("ANALYTICS_PORT", "10000"))
# NEW: let panel decide to call Binance or not (default false on Render)
PANEL_CALLS_BINANCE = os.getenv("PANEL_CALLS_BINANCE", "false").lower() == "true"

# ROI model
EQUITY_USDT = float(os.getenv("EQUITY_USDT", "1000"))
ROI_TRADES_PER_DAY = float(os.getenv("ROI_TRADES_PER_DAY", "20"))
ROI_WIN_RATE = float(os.getenv("ROI_WIN_RATE", "0.54"))
ROI_AVG_WIN_R = float(os.getenv("ROI_AVG_WIN_R", "0.9"))
ROI_AVG_LOSS_R = float(os.getenv("ROI_AVG_LOSS_R", "0.8"))
ROI_RISK_PER_TRADE_USDT = float(os.getenv("ROI_RISK_PER_TRADE_USDT", "5"))

# --------------------------
# Endpoints
# --------------------------
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/chat/completions"

if USE_TESTNET:
    BINANCE_FAPI_BASE = "https://testnet.binancefuture.com"
else:
    BINANCE_FAPI_BASE = "https://fapi.binance.com"

# --------------------------
# Helpers
# --------------------------
class BinanceHTTPError(Exception):
    pass

class LogicSkip(Exception):
    pass

def _ts_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def _hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

def as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == 'm':
        return n * 60 * 1000
    if unit == 'h':
        return n * 60 * 60 * 1000
    if unit == 'd':
        return n * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")

def pct_rank(x: np.ndarray, v: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.5
    return float((x <= v).sum()) / float(len(x))

# --------------------------
# Database adapter (SQLite local, Postgres on Render if DATABASE_URL set)
# --------------------------
import sqlite3
import contextlib

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL != ""

if USE_PG:
    import psycopg

@contextlib.contextmanager
def db_conn():
    if USE_PG:
        conn = psycopg.connect(DATABASE_URL, autocommit=True)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = sqlite3.connect("trader.db")
        try:
            yield conn
        finally:
            conn.close()

def db_init():
    with db_conn() as conn:
        cur = conn.cursor()
        if USE_PG:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS decisions(
                id SERIAL PRIMARY KEY,
                ts DOUBLE PRECISION, symbol TEXT, price DOUBLE PRECISION,
                llm_signal TEXT, confidence DOUBLE PRECISION,
                gated_signal TEXT, sl_pct DOUBLE PRECISION, tp_pct DOUBLE PRECISION,
                regime_json TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trades(
                id SERIAL PRIMARY KEY,
                ts_open DOUBLE PRECISION, symbol TEXT, side TEXT, qty DOUBLE PRECISION,
                entry_price DOUBLE PRECISION, sl_price DOUBLE PRECISION, tp_price DOUBLE PRECISION,
                ts_close DOUBLE PRECISION, close_price DOUBLE PRECISION, realized_pnl DOUBLE PRECISION, exit_reason TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS regime(
                id SERIAL PRIMARY KEY,
                ts DOUBLE PRECISION, features_json TEXT, info_json TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS account_cache(
                id SERIAL PRIMARY KEY,
                ts DOUBLE PRECISION,
                balance_usdt DOUBLE PRECISION,
                pnl24h DOUBLE PRECISION
            );""")
        else:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, symbol TEXT, price REAL,
                llm_signal TEXT, confidence REAL,
                gated_signal TEXT, sl_pct REAL, tp_pct REAL,
                regime_json TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_open REAL, symbol TEXT, side TEXT, qty REAL,
                entry_price REAL, sl_price REAL, tp_price REAL,
                ts_close REAL, close_price REAL, realized_pnl REAL, exit_reason TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS regime(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, features_json TEXT, info_json TEXT
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS account_cache(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                balance_usdt REAL,
                pnl24h REAL
            );""")
            conn.commit()

def db_insert(table: str, row: dict):
    with db_conn() as conn:
        cur = conn.cursor()
        cols = list(row.keys())
        vals = list(row.values())
        col_sql = ",".join(cols)
        if USE_PG:
            placeholders = ",".join(["%s"]*len(cols))
            cur.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders});", vals)
        else:
            placeholders = ",".join(["?"]*len(cols))
            cur.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders});", vals)
            conn.commit()

def db_update_trade_close(trade_id: int, ts_close: float, close_price: float, realized_pnl: Optional[float], reason: str):
    with db_conn() as conn:
        cur = conn.cursor()
        if USE_PG:
            cur.execute(
                "UPDATE trades SET ts_close=%s, close_price=%s, realized_pnl=%s, exit_reason=%s WHERE id=%s;",
                (ts_close, close_price, realized_pnl, reason, trade_id)
            )
        else:
            cur.execute(
                "UPDATE trades SET ts_close=?, close_price=?, realized_pnl=?, exit_reason=? WHERE id=?;",
                (ts_close, close_price, realized_pnl, reason, trade_id)
            )
            conn.commit()

def db_select_rows(sql_sqlite: str, sql_pg: str, params: tuple = ()):
    with db_conn() as conn:
        cur = conn.cursor()
        if USE_PG:
            cur.execute(sql_pg, params)
        else:
            cur.execute(sql_sqlite, params)
        return cur.fetchall()

# --------------------------
# Binance Futures client
# --------------------------
class BinanceFutures:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.ak = api_key
        self.sk = api_secret
        self.base = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.ak})

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
           retry=retry_if_exception_type(BinanceHTTPError))
    def _request(self, method: str, path: str, params: Dict[str, Any] = None, signed: bool = False):
        url = self.base + path
        params = params or {}
        if signed:
            params["timestamp"] = _ts_ms()
            params.setdefault("recvWindow", 5000)
        try:
            if method == "GET":
                resp = self.session.get(url, params=params)
            elif method == "POST":
                if signed:
                    sig = _hmac_sha256(self.sk, "&".join(f"{k}={params[k]}" for k in sorted(params.keys())))
                    resp = self.session.post(url, params={**params, "signature": sig})
                else:
                    resp = self.session.post(url, params=params)
            elif method == "DELETE":
                if signed:
                    sig = _hmac_sha256(self.sk, "&".join(f"{k}={params[k]}" for k in sorted(params.keys())))
                    resp = self.session.delete(url, params={**params, "signature": sig})
                else:
                    resp = self.session.delete(url, params=params)
            else:
                raise ValueError("Unsupported method")
        except requests.RequestException as e:
            raise BinanceHTTPError(str(e)) from e

        if resp.status_code >= 400:
            raise BinanceHTTPError(f"{resp.status_code} | {resp.text}")
        return resp.json()

    # Market data
    def klines(self, symbol: str, interval: str, limit: int = 200):
        return self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int, step_limit: int = 1500) -> List[list]:
        out: List[list] = []
        itv = interval_to_ms(interval)
        cur = start_ms
        while cur < end_ms:
            params = {
                "symbol": symbol, "interval": interval, "limit": step_limit,
                "startTime": cur, "endTime": min(cur + itv * step_limit, end_ms)
            }
            chunk = self._request("GET", "/fapi/v1/klines", params)
            if not chunk:
                break
            out.extend(chunk)
            last_close = int(chunk[-1][6])
            nxt = last_close + 1
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.05)
        return out

    def exchange_info(self, symbol: str):
        data = self._request("GET", "/fapi/v1/exchangeInfo")
        for s in data.get("symbols", []):
            if s.get("symbol") == symbol:
                return s
        raise BinanceHTTPError(f"Symbol {symbol} not found in exchangeInfo")

    # Trading
    def change_leverage(self, symbol: str, leverage: int):
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def new_order(self, **params):
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all(self, symbol: str):
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    def position_risk(self):
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)

    def open_orders(self, symbol: Optional[str] = None):
        p = {}
        if symbol:
            p["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", p, signed=True)

    # Account / PnL
    def income_history(self, startTime: Optional[int] = None, endTime: Optional[int] = None, incomeType: Optional[str] = None):
        p = {}
        if startTime: p["startTime"] = startTime
        if endTime: p["endTime"] = endTime
        if incomeType: p["incomeType"] = incomeType  # "REALIZED_PNL"
        return self._request("GET", "/fapi/v1/income", p, signed=True)

    def balances(self):
        return self._request("GET", "/fapi/v2/balance", signed=True)

# --------------------------
# Filters quantization
# --------------------------
@dataclass
class SymbolFilters:
    step_size: float
    min_qty: float
    tick_size: float

def parse_filters(ex_info: Dict[str, Any]) -> SymbolFilters:
    step_size = 0.0
    min_qty = 0.0
    tick_size = 0.0
    for f in ex_info.get("filters", []):
        ft = f.get("filterType")
        if ft == "LOT_SIZE":
            step_size = float(f["stepSize"])
            min_qty = float(f["minQty"])
        elif ft == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
    if not (step_size and tick_size):
        raise BinanceHTTPError("Missing LOT_SIZE or PRICE_FILTER in exchangeInfo")
    return SymbolFilters(step_size=step_size, min_qty=min_qty, tick_size=tick_size)

def quantize_qty(qty: float, step: float, min_qty: float) -> float:
    if qty < min_qty:
        return 0.0
    steps = math.floor(qty / step)
    return round(steps * step, 12)

def quantize_price(px: float, tick: float) -> float:
    ticks = math.floor(px / tick)
    return round(ticks * tick, 12)

# --------------------------
# DeepSeek (strict JSON)
# --------------------------
class DeepSeek:
    def __init__(self, api_key: str):
        self.ak = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.ak}",
            "Content-Type": "application/json",
        })

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
           retry=retry_if_exception_type(requests.RequestException))
    def decide(self, features: Dict[str, float]) -> Dict[str, Any]:
        sys = (
            "You are a trading signal engine. Output STRICT JSON with keys: "
            "signal (one of 'long','short','flat'), confidence (0..1 float), "
            "sl_pct (positive float like 0.01), tp_pct (positive float). "
            "No prose. No code fences."
        )
        usr = (
            "Decide next action from these normalized features including cross-asset regime data. "
            "Favor trades aligned with BTC trend; be conservative during high BTC volatility or weak correlation. "
            "JSON ONLY.\n"
            f"{json.dumps(features, separators=(',', ':'))}"
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            "temperature": 0.15,
            "max_tokens": 150,
            "response_format": {"type": "json_object"},
        }
        r = self.session.post(DEEPSEEK_BASE + DEEPSEEK_CHAT_PATH, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        sig = parsed.get("signal", "flat")
        conf = float(parsed.get("confidence", 0.0))
        slp = float(parsed.get("sl_pct", 0.01))
        tpp = float(parsed.get("tp_pct", 0.02))
        if sig not in ("long", "short", "flat"):
            sig = "flat"
        return {"signal": sig, "confidence": conf, "sl_pct": abs(slp), "tp_pct": abs(tpp)}

# --------------------------
# Short-term features
# --------------------------
def build_features(klines: list) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cols = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tqv", "i"]
    df = pd.DataFrame(klines, columns=cols[:len(klines[0])])
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    df["ret1"] = df["c"].pct_change().fillna(0)
    df["hl_range"] = (df["h"] - df["l"]) / df["o"].replace(0, np.nan)
    df["ema_fast"] = ema(df["c"], 9)
    df["ema_slow"] = ema(df["c"], 21)
    df["rsi14"] = rsi(df["c"], 14)
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["c"]
    df["vol_z"] = (df["v"] - df["v"].rolling(30).mean()) / (df["v"].rolling(30).std() + 1e-9)
    latest = df.iloc[-1]
    feat = {
        "ret1": float(latest["ret1"]),
        "hl_range": float(latest["hl_range"]),
        "ema_diff": float(latest["ema_diff"]),
        "rsi14": float(latest["rsi14"]),
        "vol_z": float(latest["vol_z"]),
        "price": float(latest["c"]),
    }
    for k, v in list(feat.items()):
        if not np.isfinite(v):
            feat[k] = 0.0
    return df, feat

# --------------------------
# Yearly cross-asset regime
# --------------------------
@dataclass
class RegimeSnapshot:
    ts: float
    features: Dict[str, float]
    info: Dict[str, Any]

class RegimeAnalyzer:
    def __init__(self, client: BinanceFutures, lock: Optional[Lock] = None):
        self.b = client
        self.lock = lock or Lock()
        self.snapshot: Optional[RegimeSnapshot] = None
        self.run_flag = True
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.run_flag = False

    def _loop(self):
        while self.run_flag:
            try:
                snap = self._compute_once()
                with self.lock:
                    self.snapshot = snap
                db_insert("regime", {
                    "ts": snap.ts,
                    "features_json": json.dumps(snap.features),
                    "info_json": json.dumps(snap.info),
                })
                print(f"[Regime] refreshed @ {datetime.utcnow().isoformat(timespec='seconds')}Z")
            except Exception as e:
                print(f"[Regime] error: {e}")
            time.sleep(REGIME_REFRESH_MINUTES * 60)

    def get_snapshot(self) -> Optional[RegimeSnapshot]:
        with self.lock:
            return self.snapshot

    def _fetch_year(self, symbol: str, interval: str) -> pd.DataFrame:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=REGIME_LOOKBACK_DAYS)
        kl = self.b.klines_range(
            symbol=symbol, interval=interval,
            start_ms=int(start.timestamp() * 1000),
            end_ms=int(end.timestamp() * 1000),
            step_limit=1500
        )
        if not kl or len(kl) < 100:
            raise LogicSkip(f"Not enough history for {symbol}")
        cols = ["t", "o", "h", "l", "c", "v", "ct"]
        df = pd.DataFrame(kl, columns=cols[:len(kl[0])])
        for col in ("o", "h", "l", "c", "v"):
            df[col] = df[col].astype(float)
        df["dt"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
        df.set_index("dt", inplace=True)
        return df

    def _compute_once(self) -> RegimeSnapshot:
        btc = self._fetch_year("BTCUSDT", REGIME_INTERVAL)
        alt = self._fetch_year("1000PEPEUSDT", REGIME_INTERVAL)

        joined = btc[["c"]].rename(columns={"c": "btc_c"}).join(
            alt[["c"]].rename(columns={"c": "alt_c"}), how="inner"
        ).dropna()
        joined["btc_r"] = joined["btc_c"].pct_change()
        joined["alt_r"] = joined["alt_c"].pct_change()

        btc_close = joined["btc_c"]
        btc_ema50 = ema(btc_close, 50)
        btc_ema200 = ema(btc_close, 200)
        btc_trend_raw = (btc_ema50 - btc_ema200) / btc_close
        btc_trend_state = np.sign(btc_trend_raw.iloc[-1])

        win = int(24 * 30)  # 30 days of 1h
        btc_vol = joined["btc_r"].rolling(win).std()
        vol_latest = float(btc_vol.iloc[-1])
        vol_hist = btc_vol.values
        vol_pct = pct_rank(vol_hist[np.isfinite(vol_hist)], vol_latest)

        w7 = 24 * 7
        sub = joined.iloc[-w7:].dropna()
        if len(sub) < 10 or sub["btc_r"].var() == 0:
            corr7 = 0.0
            beta7 = 0.0
        else:
            corr7 = float(sub["alt_r"].corr(sub["btc_r"]))
            beta7 = float(np.cov(sub["alt_r"], sub["btc_r"])[0, 1] / (sub["btc_r"].var() + 1e-12))

        alt_cum = (1 + sub["alt_r"]).prod() - 1 if len(sub) else 0.0
        btc_cum = (1 + sub["btc_r"]).prod() - 1 if len(sub) else 0.0
        rel_str = float(alt_cum - beta7 * btc_cum)

        tmp = pd.DataFrame({"c": joined["btc_c"]})
        tmp["rsi14"] = rsi(tmp["c"], 14)
        btc_rsi = float(tmp["rsi14"].iloc[-1])

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
            "regime_is_chop": float(is_chop)
        }
        info = {
            "btc_price": float(btc_close.iloc[-1]),
            "alt_price": float(joined['alt_c'].iloc[-1]),
            "samples_1h": int(len(joined)),
            "window_corr_hours": int(w7)
        }
        return RegimeSnapshot(ts=time.time(), features=feat, info=info)

# --------------------------
# Account cache writer (worker → DB; panel reads it)
# --------------------------
class AccountCacheWriter:
    def __init__(self, bclient: BinanceFutures, interval_sec: int = 300):
        self.b = bclient
        self.interval = max(60, interval_sec)
        self._stop = False
        self.th = Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                bal = None
                pnl = None
                try:
                    # balance
                    res = self.b.balances()
                    usdt = [x for x in res if x.get("asset") == "USDT"]
                    if usdt:
                        bal = float(usdt[0].get("balance", 0.0))
                except Exception as e:
                    print(f"[AcctCache] balance err: {e}")

                try:
                    # 24h realized pnl
                    since = int((datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000)
                    inc = self.b.income_history(startTime=since, incomeType="REALIZED_PNL")
                    total = 0.0
                    for x in inc:
                        total += float(x.get("income", 0.0))
                    pnl = total
                except Exception as e:
                    print(f"[AcctCache] pnl24h err: {e}")

                db_insert("account_cache", {
                    "ts": time.time(),
                    "balance_usdt": bal if bal is not None else None,
                    "pnl24h": pnl if pnl is not None else None
                })
            except Exception as e:
                print(f"[AcctCache] loop err: {e}")
            time.sleep(self.interval)

# --------------------------
# Trading logic
# --------------------------
class AutoTrader:
    def __init__(self, b: BinanceFutures, d: DeepSeek, symbol: str):
        self.b = b
        self.d = d
        self.symbol = symbol
        self.ex_info = self.b.exchange_info(self.symbol)
        self.filters = parse_filters(self.ex_info)
        self.position_open_time = None

        self.active_trade_id: Optional[int] = None
        self.active_entry_price: Optional[float] = None
        self.active_side: Optional[str] = None
        self.active_qty: Optional[float] = None

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
            try:
                self.b.cancel_all(self.symbol)
            except Exception as e:
                print(f"[Cleanup] cancel_all: {e}")

    def _round_qty_for_notional(self, price: float, notional_usdt: float) -> float:
        raw = notional_usdt / price
        return quantize_qty(raw, self.filters.step_size, self.filters.min_qty)

    def _dynamic_notional(self, base_notional: float, regime_feat: Dict[str, float]) -> float:
        vol_pct = float(regime_feat.get("btc_vol_pct", 0.5))
        scale = SIZE_SCALE_MAX - (SIZE_SCALE_MAX - SIZE_SCALE_MIN) * vol_pct
        scaled = base_notional * float(np.clip(scale, SIZE_SCALE_MIN, SIZE_SCALE_MAX))
        return max(5.0, scaled)

    def _hybrid_gate(self, desired_signal: str, regime_feat: Dict[str, float]) -> str:
        if not USE_HYBRID_GATES:
            return desired_signal
        trend = float(regime_feat.get("btc_trend_state", 0.0))
        volp = float(regime_feat.get("btc_vol_pct", 0.5))
        corr = float(regime_feat.get("corr7", 0.0))
        chop = float(regime_feat.get("regime_is_chop", 0.0))
        if chop >= 0.5 and (volp >= 0.55 or abs(corr) <= 0.15):
            return "flat"
        if trend < 0 and volp >= 0.6 and desired_signal == "long":
            return "flat"
        if trend > 0 and corr >= 0.2 and desired_signal == "short":
            return "flat"
        return desired_signal

    def _place_entry_and_exits(self, signal: str, price: float, sl_pct: float, tp_pct: float, notional_override: Optional[float] = None):
        use_notional = notional_override if notional_override is not None else NOTIONAL_PER_TRADE_USDT
        qty = self._round_qty_for_notional(price, use_notional)
        if qty <= 0:
            raise LogicSkip("Qty below LOT_SIZE; increase NOTIONAL_PER_TRADE_USDT")

        side = "BUY" if signal == "long" else "SELL"
        order = self.b.new_order(symbol=self.symbol, side=side, type="MARKET", quantity=f"{qty}")
        print(f"[Entry] {side} {qty} @ market → id={order.get('orderId')}")
        self.active_entry_price = price
        self.active_side = side
        self.active_qty = qty
        self.position_open_time = time.time()

        if signal == "long":
            sl_price = quantize_price(price * (1 - sl_pct), self.filters.tick_size)
            tp_price = quantize_price(price * (1 + tp_pct), self.filters.tick_size)
            sl_side = "SELL"; tp_side = "SELL"
        else:
            sl_price = quantize_price(price * (1 + sl_pct), self.filters.tick_size)
            tp_price = quantize_price(price * (1 - tp_pct), self.filters.tick_size)
            sl_side = "BUY"; tp_side = "BUY"

        self.b.new_order(symbol=self.symbol, side=sl_side, type="STOP_MARKET",
                         stopPrice=f"{sl_price}", closePosition="true", workingType=RISK_WORKING_TYPE)
        self.b.new_order(symbol=self.symbol, side=tp_side, type="TAKE_PROFIT_MARKET",
                         stopPrice=f"{tp_price}", closePosition="true", workingType=RISK_WORKING_TYPE)
        print(f"[Exits] SL {sl_side} @{sl_price} | TP {tp_side} @{tp_price}")

        db_insert("trades", {
            "ts_open": time.time(), "symbol": self.symbol, "side": side, "qty": qty,
            "entry_price": price, "sl_price": sl_price, "tp_price": tp_price,
            "ts_close": None, "close_price": None, "realized_pnl": None, "exit_reason": None
        })
        rows = db_select_rows(
            "SELECT id FROM trades ORDER BY id DESC LIMIT 1;",
            "SELECT id FROM trades ORDER BY id DESC LIMIT 1;",
            ()
        )
        self.active_trade_id = int(rows[0][0]) if rows else None

    def _close_if_timeout(self, current_price: float):
        if self.position_open_time and time.time() - self.position_open_time > MAX_OPEN_SECONDS:
            amt = self._flat_position_amt()
            if abs(amt) > 0:
                side = "SELL" if amt > 0 else "BUY"
                self.b.cancel_all(self.symbol)
                self.b.new_order(symbol=self.symbol, side=side, type="MARKET",
                                 quantity=f"{abs(amt)}")
                print("[Panic] Closed due to MAX_OPEN_SECONDS")
                self._finalize_trade("timeout", current_price)

    def _finalize_trade(self, reason: str, close_price: float):
        if self.active_trade_id is None:
            return
        realized = None
        try:
            since = int((datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000)
            inc = self.b.income_history(startTime=since, incomeType="REALIZED_PNL")
            realized_sum = 0.0
            for x in inc:
                if x.get("symbol") == self.symbol:
                    realized_sum += float(x.get("income", 0.0))
            realized = realized_sum
        except Exception as e:
            print(f"[PnL] income fallback: {e}")
        db_update_trade_close(self.active_trade_id, time.time(), close_price, realized, reason)
        self.active_trade_id = None
        self.active_entry_price = None
        self.active_side = None
        self.active_qty = None
        self.position_open_time = None

    def step(self):
        kl = self.b.klines(self.symbol, INTERVAL, limit=KLIMIT)
        if not kl or len(kl) < 50:
            raise LogicSkip("Not enough bars")
        _, feat = build_features(kl)
        price = feat["price"]

        self._close_if_timeout(price)

        if abs(self._flat_position_amt()) > 0:
            raise LogicSkip("Position active; waiting")

        snap = self.regime.get_snapshot()
        if not snap:
            raise LogicSkip("Regime snapshot not ready")

        all_feat = {**feat}
        for k, v in snap.features.items():
            all_feat[f"X_{k}"] = float(v)

        decision = self.d.decide(all_feat)
        sig = decision["signal"]
        conf = decision["confidence"]
        slp = max(0.001, decision["sl_pct"])
        tpp = max(0.001, decision["tp_pct"])

        gated_sig = self._hybrid_gate(sig, snap.features)

        db_insert("decisions", {
            "ts": time.time(), "symbol": self.symbol, "price": price,
            "llm_signal": sig, "confidence": conf, "gated_signal": gated_sig,
            "sl_pct": slp, "tp_pct": tpp, "regime_json": json.dumps(snap.features)
        })

        print(f"[Decision] LLM={sig}({conf:.3f}) → Gate={gated_sig} | sl={slp:.3f} tp={tpp:.3f} | px={price} "
              f"| trend={snap.features.get('btc_trend_state')} volPct={snap.features.get('btc_vol_pct'):.2f} "
              f"| corr7={snap.features.get('corr7'):.2f} beta7={snap.features.get('beta7'):.2f} "
              f"| RS7={snap.features.get('alt_rel_strength_7d'):.4f} chop={snap.features.get('regime_is_chop')}")

        if gated_sig == "flat" or conf < CONFIDENCE_THRESHOLD:
            raise LogicSkip("No trade (confidence low / gate flat)")

        dyn_notional = self._dynamic_notional(NOTIONAL_PER_TRADE_USDT, snap.features)
        self._place_entry_and_exits(gated_sig, price, slp, tpp, notional_override=dyn_notional)

        time.sleep(1)
        self._cleanup_if_flat()
        if abs(self._flat_position_amt()) < 1e-12 and self.active_trade_id is not None:
            self._finalize_trade("instant_exit", price)

    def loop(self):
        print(f"[Bot] Testnet={USE_TESTNET} | Symbol={self.symbol} | Lev={LEVERAGE} | Notional={NOTIONAL_PER_TRADE_USDT} USDT")
        print(f"[Regime] Lookback={REGIME_LOOKBACK_DAYS}d @ {REGIME_INTERVAL} | Refresh={REGIME_REFRESH_MINUTES}m | Gates={USE_HYBRID_GATES}")
        print("[Bot] Running. Ctrl+C to stop.")
        while True:
            try:
                self.step()
            except LogicSkip as e:
                print(f"[Skip] {e}")
            except BinanceHTTPError as e:
                print(f"[Binance] {e}")
            except requests.RequestException as e:
                print(f"[HTTP] {e}")
            except KeyboardInterrupt:
                print("Bye.")
                break
            except Exception as e:
                print(f"[Error] {type(e).__name__}: {e}")
            time.sleep(DECISION_COOLDOWN_SEC)

# --------------------------
# Analytics Panel (ASGI app builder + optional local runner)
# --------------------------
def build_panel_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    # Ensure tables exist (web and worker are separate containers)
    try:
        db_init()
    except Exception as e:
        print(f"[Panel] db_init warning: {e}")

    app = FastAPI()

    DASH = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>AutoTrader Panel</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;color:#111;background:#fafafa;margin:0;padding:20px}
.card{background:#fff;border-radius:16px;box-shadow:0 1px 8px rgba(0,0,0,.06);padding:16px;margin-bottom:16px}
h1{margin:0 0 12px 0} table{width:100%;border-collapse:collapse} th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}
.badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;background:#eef}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
</style>
</head><body>
<h1>DeepSeek × Binance AutoTrader</h1>
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
<script>
async function load(){
  const s = await fetch('/api/status').then(r=>r.json());
  const r = await fetch('/api/roi').then(r=>r.json());
  const re= await fetch('/api/resources').then(r=>r.json());
  const d = await fetch('/api/decisions?limit=20').then(r=>r.json());
  const g = await fetch('/api/regime').then(r=>r.json());
  document.getElementById('status').innerHTML =
    `<div><span class="badge">Symbol</span> ${s.symbol}</div>
     <div><span class="badge">Testnet</span> ${s.testnet}</div>
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
}
setInterval(load, 4000); load();
</script>
</body></html>
    """

    def _balances_pnl_from_cache():
        rows = db_select_rows(
            "SELECT balance_usdt,pnl24h FROM account_cache ORDER BY id DESC LIMIT 1;",
            "SELECT balance_usdt,pnl24h FROM account_cache ORDER BY id DESC LIMIT 1;",
            ()
        )
        if rows:
            b, p = rows[0]
            return (None if b is None else float(b)), (None if p is None else float(p))
        return (None, None)

    def _balances_live():
        try:
            b = BinanceFutures(BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_FAPI_BASE)
            res = b.balances()
            usdt = [x for x in res if x.get("asset") == "USDT"]
            if usdt:
                return float(usdt[0].get("balance", 0.0))
        except Exception as e:
            print(f"[Panel] balances live err: {e}")
        return None

    def _pnl24h_live():
        try:
            b = BinanceFutures(BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_FAPI_BASE)
            since = int((datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000)
            inc = b.income_history(startTime=since, incomeType="REALIZED_PNL")
            total = 0.0
            for x in inc:
                total += float(x.get("income", 0.0))
            return total
        except Exception as e:
            print(f"[Panel] pnl24h live err: {e}")
            return None

    def _roi_model():
        ev_trade = ROI_WIN_RATE * (ROI_AVG_WIN_R * ROI_RISK_PER_TRADE_USDT) - (1 - ROI_WIN_RATE) * (ROI_AVG_LOSS_R * ROI_RISK_PER_TRADE_USDT)
        daily_pnl = ev_trade * ROI_TRADES_PER_DAY
        daily_roi = daily_pnl / max(1.0, EQUITY_USDT)
        return {
            "expected_daily_pnl_usdt": daily_pnl,
            "expected_daily_roi": daily_roi,
            "trades_per_day": ROI_TRADES_PER_DAY,
            "win_rate": ROI_WIN_RATE,
            "avg_win_r": ROI_AVG_WIN_R,
            "avg_loss_r": ROI_AVG_LOSS_R,
            "risk_per_trade": ROI_RISK_PER_TRADE_USDT,
        }

    def _resources():
        year_bars = 8760
        cols = 10
        bytes_per = year_bars * cols * 8 * 2
        mem_mb = bytes_per / (1024**2)
        cpu_needs = "1 vCPU / 512MB is fine; use 2 vCPU / 1–2GB for more symbols or 10–15s loop."
        return {"year_bars": year_bars, "mem_mb": mem_mb, "cpu_needs": cpu_needs}

    app.DASH = DASH

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(DASH)

    @app.get("/api/decisions")
    def api_decisions(limit: int = 50):
        try:
            rows = db_select_rows(
                "SELECT ts,price,llm_signal,confidence,gated_signal FROM decisions ORDER BY id DESC LIMIT ?;",
                "SELECT ts,price,llm_signal,confidence,gated_signal FROM decisions ORDER BY id DESC LIMIT %s;",
                (limit,)
            )
            rows = [{"ts": r[0], "price": r[1], "llm_signal": r[2], "confidence": r[3], "gated_signal": r[4]} for r in rows]
            return {"rows": rows}
        except Exception as e:
            print(f"[Panel] decisions query: {e}")
            return {"rows": []}

    @app.get("/api/regime")
    def api_regime():
        try:
            rows = db_select_rows(
                "SELECT ts,features_json,info_json FROM regime ORDER BY id DESC LIMIT 1;",
                "SELECT ts,features_json,info_json FROM regime ORDER BY id DESC LIMIT 1;",
                ()
            )
            if not rows:
                return {}
            row = rows[0]
            return {"ts": row[0], "features": json.loads(row[1]), "info": json.loads(row[2])}
        except Exception as e:
            print(f"[Panel] regime query: {e}")
            return {}

    @app.get("/api/status")
    def api_status():
        # Prefer cache written by worker; optionally use live if PANEL_CALLS_BINANCE=true
        bal, pnl = _balances_pnl_from_cache()
        if PANEL_CALLS_BINANCE and (bal is None or pnl is None):
            bal = _balances_live() if bal is None else bal
            pnl = _pnl24h_live() if pnl is None else pnl
        return {
            "symbol": SYMBOL,
            "testnet": USE_TESTNET,
            "leverage": LEVERAGE,
            "notional": NOTIONAL_PER_TRADE_USDT,
            "balance_usdt": bal,
            "pnl24h": pnl,
        }

    @app.get("/api/roi")
    def api_roi():
        return _roi_model()

    @app.get("/api/resources")
    def api_resources():
        return _resources()

    return app


def start_panel():
    """Local dev convenience. On Render, use panel.py (panel:app)."""
    import uvicorn
    app = build_panel_app()
    uvicorn.run(app, host=ANALYTICS_HOST, port=ANALYTICS_PORT, log_level="warning")

# --------------------------
# Main (Worker entrypoint)
# --------------------------
def main():
    if not (DEEPSEEK_API_KEY and BINANCE_API_KEY and BINANCE_API_SECRET):
        print("Missing keys (DEEPSEEK_API_KEY, BINANCE_API_KEY, BINANCE_API_SECRET).")
        return

    db_init()  # ensure tables

    if ANALYTICS_ENABLED:
        # local dev only
        Thread(target=start_panel, daemon=True).start()

    b = BinanceFutures(BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_FAPI_BASE)
    d = DeepSeek(DEEPSEEK_API_KEY)

    # start account cache writer (panel will read this)
    AccountCacheWriter(b, interval_sec=300)

    AutoTrader(b, d, SYMBOL).loop()

if __name__ == "__main__":
    main()
