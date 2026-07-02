#!/usr/bin/env python3
"""
Falling Knife Trading Agent

A CLI tool inspired by Michael Burry's approach to scaling into dropping assets
without catching a falling knife prematurely. Uses volume turnover relative to
shares outstanding to gauge whether the shareholder base has sufficiently rotated.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional, Tuple

import requests
import yfinance as yf
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

ASSET_TYPES = ("Old Guard", "Standard", "High-Growth")

TURNOVER_THRESHOLDS: dict[str, Tuple[float, float]] = {
    "Old Guard": (1.5, 3.0),
    "Standard": (3.0, 5.0),
    "High-Growth": (5.0, 10.0),
}

INITIAL_DROP_THRESHOLD = 20.0  # percent drop from initial_price triggers DCA buy
HISTORY_PERIOD = "1y"

console = Console()

YFINANCE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def create_yfinance_session() -> requests.Session:
    """Build a requests session with a browser User-Agent for Yahoo Finance."""
    session = requests.Session()
    session.headers.update({"User-Agent": YFINANCE_USER_AGENT})
    return session


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def fetch_ticker_data(symbol):
    """Return a yfinance Ticker object for the given symbol with a custom session."""
    session = create_yfinance_session()
    return yf.Ticker(symbol.upper(), session=session)


def get_shares_outstanding(ticker: yf.Ticker) -> Optional[float]:
    """
    Attempt to resolve shares outstanding from several yfinance info keys.
    ETFs and some tickers may use different field names or omit the value.
    """
    info: dict[str, Any] = ticker.info or {}

    keys = (
        "sharesOutstanding",
        "impliedSharesOutstanding",
        "floatShares",
    )

    for key in keys:
        value = info.get(key)
        if value is not None and isinstance(value, (int, float)) and value > 0:
            return float(value)

    return None


def get_current_price(ticker: yf.Ticker, history) -> Optional[float]:
    """Derive the most recent tradable price from history or fast info."""
    if history is not None and not history.empty:
        return float(history["Close"].iloc[-1])

    info = ticker.info or {}
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        value = info.get(key)
        if value is not None:
            return float(value)

    return None


def find_recent_peak(history) -> Tuple[Optional[float], Optional[datetime]]:
    """
    Find the all-time high (within the history window) and the date of the
  most recent occurrence of that high — the point from which the current drop
    is measured.
    """
    if history is None or history.empty or "High" not in history.columns:
        return None, None

    peak_price = float(history["High"].max())
    peak_rows = history[history["High"] == peak_price]
    peak_date = peak_rows.index[-1]

    # Normalize timezone-aware timestamps to naive datetimes for display.
    if hasattr(peak_date, "to_pydatetime"):
        peak_date = peak_date.to_pydatetime()
    if hasattr(peak_date, "tzinfo") and peak_date.tzinfo is not None:
        peak_date = peak_date.replace(tzinfo=None)

    return peak_price, peak_date


def accumulated_volume_since(history, peak_date: datetime) -> Optional[float]:
    """Sum daily volume from peak_date through the latest bar."""
    if history is None or history.empty or "Volume" not in history.columns:
        return None

    # Align peak_date to the history index (may be timezone-aware).
    idx = history.index
    if hasattr(idx, "tz") and idx.tz is not None:
        peak_ts = peak_date
        if peak_ts.tzinfo is None:
            peak_ts = peak_ts.replace(tzinfo=idx.tz)
        mask = idx >= peak_ts
    else:
        mask = idx >= peak_date

    segment = history.loc[mask, "Volume"]
    if segment.empty:
        return 0.0

    return float(segment.sum())


def pct_drop(from_price: float, to_price: float) -> float:
    """Percentage decline from from_price to to_price."""
    if from_price <= 0:
        return 0.0
    return ((from_price - to_price) / from_price) * 100.0


HealthStatus = Literal["green", "red", "neutral", "unknown"]


@dataclass
class FundamentalHealth:
    """Fundamental metrics used to assess balance-sheet and earnings health."""

    total_cash: Optional[float]
    total_debt: Optional[float]
    debt_to_cash_ratio: Optional[float]
    debt_to_cash_status: HealthStatus
    operating_cashflow: Optional[float]
    cash_runway_years: Optional[float]
    cash_runway_status: HealthStatus
    cash_runway_note: str
    current_ratio: Optional[float]
    current_ratio_status: HealthStatus
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    trailing_pe_label: str
    forward_pe_label: str
    revenue_growth_pct: Optional[float]
    revenue_growth_label: str
    revenue_growth_emphasized: bool
    trend_label: str
    trend_status: HealthStatus


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_emoji(status: HealthStatus) -> str:
    if status == "green":
        return "🟢"
    if status == "red":
        return "🔴"
    return "—"


def _quarterly_series(ticker: yf.Ticker, row_names: tuple[str, ...]) -> Optional[list[float]]:
    """Return the most recent quarterly values for the first matching financial row."""
    try:
        quarterly = ticker.quarterly_financials
        if quarterly is None or quarterly.empty:
            quarterly = ticker.quarterly_income_stmt
        if quarterly is None or quarterly.empty:
            return None

        for name in row_names:
            if name in quarterly.index:
                values = quarterly.loc[name].dropna().sort_index()
                if not values.empty:
                    return [float(v) for v in values.tail(4).tolist()]
    except Exception:
        return None
    return None


def extract_fundamental_health(ticker: yf.Ticker) -> FundamentalHealth:
    """Pull cash, liquidity, and earnings trend metrics from yfinance."""
    info: dict[str, Any] = ticker.info or {}

    total_cash = _safe_float(info.get("totalCash"))
    total_debt = _safe_float(info.get("totalDebt"))
    operating_cashflow = _safe_float(info.get("operatingCashflow"))
    current_ratio = _safe_float(info.get("currentRatio"))
    trailing_pe_raw = _safe_float(info.get("trailingPE"))
    forward_pe_raw = _safe_float(info.get("forwardPE"))
    trailing_pe = (
        trailing_pe_raw if trailing_pe_raw is not None and trailing_pe_raw > 0 else None
    )
    forward_pe = (
        forward_pe_raw if forward_pe_raw is not None and forward_pe_raw > 0 else None
    )

    debt_to_cash_ratio: Optional[float] = None
    debt_to_cash_status: HealthStatus = "unknown"
    if total_debt is not None and total_cash is not None and total_cash > 0:
        debt_to_cash_ratio = total_debt / total_cash
        if debt_to_cash_ratio <= 1.0:
            debt_to_cash_status = "green"
        elif debt_to_cash_ratio >= 2.0:
            debt_to_cash_status = "red"
        else:
            debt_to_cash_status = "neutral"
    elif total_debt is not None and (total_cash is None or total_cash <= 0):
        debt_to_cash_status = "red"

    is_profitable = trailing_pe is not None or forward_pe is not None
    trailing_pe_label = f"{trailing_pe:.2f}" if trailing_pe is not None else (
        "N/A (Not Profitable)" if not is_profitable else "N/A"
    )
    forward_pe_label = f"{forward_pe:.2f}" if forward_pe is not None else (
        "N/A (Not Profitable)" if not is_profitable else "N/A"
    )

    revenue_growth_pct: Optional[float] = None
    revenue_growth_from_info = _safe_float(info.get("revenueGrowth"))
    if revenue_growth_from_info is not None:
        revenue_growth_pct = revenue_growth_from_info * 100.0

    revenue_values = _quarterly_series(
        ticker, ("Total Revenue", "Revenue", "Operating Revenue")
    )
    if revenue_growth_pct is None and revenue_values and len(revenue_values) >= 2:
        oldest, latest = revenue_values[0], revenue_values[-1]
        if oldest != 0:
            revenue_growth_pct = ((latest - oldest) / abs(oldest)) * 100.0

    revenue_growth_emphasized = not is_profitable
    if revenue_growth_pct is not None:
        prefix = "★ Revenue Growth (key metric): " if revenue_growth_emphasized else "Revenue Growth: "
        revenue_growth_label = f"{prefix}{revenue_growth_pct:+.1f}%"
    else:
        revenue_growth_label = "N/A"

    # Cash runway: meaningful when the company is burning cash (negative OCF).
    cash_runway_years: Optional[float] = None
    cash_runway_status: HealthStatus = "unknown"
    cash_runway_note = "Insufficient data"

    if total_cash is not None and operating_cashflow is not None:
        if operating_cashflow >= 0:
            cash_runway_status = "green"
            cash_runway_note = "Cash flow positive — no burn runway needed"
        elif total_cash <= 0:
            cash_runway_status = "red"
            cash_runway_note = "No cash buffer with negative operating cash flow"
            cash_runway_years = 0.0
        else:
            cash_runway_years = total_cash / abs(operating_cashflow)
            if cash_runway_years >= 1.5:
                cash_runway_status = "green"
                cash_runway_note = f"{cash_runway_years:.1f} years of runway"
            elif cash_runway_years < 1.0:
                cash_runway_status = "red"
                cash_runway_note = (
                    f"{cash_runway_years:.1f} years — risk of dilution"
                )
            else:
                cash_runway_status = "neutral"
                cash_runway_note = f"{cash_runway_years:.1f} years of runway"
    elif total_cash is not None or operating_cashflow is not None:
        cash_runway_note = "Partial data — runway could not be calculated"

    # Current ratio thresholds.
    current_ratio_status: HealthStatus = "unknown"
    if current_ratio is not None:
        if current_ratio > 2.0:
            current_ratio_status = "green"
        elif current_ratio < 1.0:
            current_ratio_status = "red"
        else:
            current_ratio_status = "neutral"

    # Revenue or EPS trend across recent quarters.
    trend_label = "N/A"
    trend_status: HealthStatus = "unknown"

    eps_values = _quarterly_series(
        ticker, ("Diluted EPS", "Basic EPS", "Net Income")
    )

    if revenue_values and len(revenue_values) >= 2:
        oldest, latest = revenue_values[0], revenue_values[-1]
        if oldest != 0:
            change_pct = ((latest - oldest) / abs(oldest)) * 100.0
            trend_label = (
                f"Revenue trend ({len(revenue_values)}Q): "
                f"{_fmt_large_number(oldest)} → {_fmt_large_number(latest)} "
                f"({change_pct:+.1f}%)"
            )
            trend_status = (
                "green" if change_pct > 5 else "red" if change_pct < -5 else "neutral"
            )
    elif eps_values and len(eps_values) >= 2:
        oldest, latest = eps_values[0], eps_values[-1]
        if oldest != 0:
            change_pct = ((latest - oldest) / abs(oldest)) * 100.0
            trend_label = (
                f"EPS trend ({len(eps_values)}Q): "
                f"${oldest:.2f} → ${latest:.2f} ({change_pct:+.1f}%)"
            )
            trend_status = (
                "green" if change_pct > 5 else "red" if change_pct < -5 else "neutral"
            )
        else:
            trend_label = (
                f"EPS trend ({len(eps_values)}Q): "
                f"${eps_values[0]:.2f} → ${eps_values[-1]:.2f}"
            )
            trend_status = "green" if eps_values[-1] > eps_values[0] else "red"

    return FundamentalHealth(
        total_cash=total_cash,
        total_debt=total_debt,
        debt_to_cash_ratio=debt_to_cash_ratio,
        debt_to_cash_status=debt_to_cash_status,
        operating_cashflow=operating_cashflow,
        cash_runway_years=cash_runway_years,
        cash_runway_status=cash_runway_status,
        cash_runway_note=cash_runway_note,
        current_ratio=current_ratio,
        current_ratio_status=current_ratio_status,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        trailing_pe_label=trailing_pe_label,
        forward_pe_label=forward_pe_label,
        revenue_growth_pct=revenue_growth_pct,
        revenue_growth_label=revenue_growth_label,
        revenue_growth_emphasized=revenue_growth_emphasized,
        trend_label=trend_label,
        trend_status=trend_status,
    )


@dataclass
class MarketSentiment:
    """Analyst consensus and simple technical trend indicators."""

    recommendation_key: Optional[str]
    recommendation_label: str
    recommendation_style: str
    analyst_count: Optional[int]
    target_mean_price: Optional[float]
    target_vs_current_pct: Optional[float]
    target_label: str
    target_style: str
    sma_50: Optional[float]
    technical_label: str
    technical_style: str
    technical_emoji: str
    volume_control_label: str
    volume_control_style: str


VolumeDayControl = Literal["buyers", "sellers", "indecision"]


def _close_range_position(high: float, low: float, close: float) -> Optional[float]:
    """Return where close sits within the day's range (0 = low, 1 = high)."""
    spread = high - low
    if spread <= 0:
        return None
    return (close - low) / spread


def _classify_volume_day(high: float, low: float, close: float) -> VolumeDayControl:
    position = _close_range_position(high, low, close)
    if position is None:
        return "indecision"
    if position >= 0.65:
        return "buyers"
    if position <= 0.35:
        return "sellers"
    return "indecision"


def analyze_volume_control(history) -> tuple[str, str]:
    """
    Classify buyer/seller control over the last 5 trading days using
    close position within the daily range and volume spikes.
    """
    required_cols = ("High", "Low", "Close", "Volume")
    if history is None or history.empty:
        return "N/A", "dim"

    if any(col not in history.columns for col in required_cols):
        return "N/A", "dim"

    recent = history.dropna(subset=list(required_cols)).tail(5)
    if recent.empty:
        return "N/A", "dim"

    counts = {"buyers": 0, "sellers": 0, "indecision": 0}
    spike_absorption_days = 0
    mean_volume = float(recent["Volume"].mean()) if len(recent) > 0 else 0.0
    spike_threshold = mean_volume * 1.5 if mean_volume > 0 else 0.0

    for _, row in recent.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        volume = float(row["Volume"])
        control = _classify_volume_day(high, low, close)
        counts[control] += 1

        if spike_threshold > 0 and volume >= spike_threshold and control == "indecision":
            spike_absorption_days += 1

    total_days = len(recent)
    latest_row = recent.iloc[-1]
    latest_control = _classify_volume_day(
        float(latest_row["High"]),
        float(latest_row["Low"]),
        float(latest_row["Close"]),
    )
    latest_volume = float(latest_row["Volume"])
    latest_spike_absorption = (
        spike_threshold > 0
        and latest_volume >= spike_threshold
        and latest_control == "indecision"
    )

    if latest_spike_absorption or spike_absorption_days > 0:
        return "Volume Spiked with Seller Absorption 🟡", "yellow"

    buyers = counts["buyers"]
    sellers = counts["sellers"]
    indecision = counts["indecision"]

    if buyers > sellers and buyers >= indecision:
        return f"{buyers}/{total_days} Days: Buyers Control 🟢", "green"
    if sellers > buyers and sellers >= indecision:
        return f"{sellers}/{total_days} Days: Sellers Control 🔴", "red"
    if indecision >= buyers and indecision >= sellers:
        return (
            f"{indecision}/{total_days} Days: Absorption / Indecision (50/50) 🟡",
            "yellow",
        )

    return (
        f"{buyers}B / {sellers}S / {indecision} Indecision (mixed)",
        "white",
    )


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_recommendation(key: Optional[str]) -> tuple[str, str]:
    """Return a display label and Rich color style for analyst recommendation."""
    if not key:
        return "N/A", "dim"

    normalized = key.lower().replace("_", " ").replace("-", " ")
    label = normalized.title()

    if key in ("strong_buy", "buy"):
        return label, "green"
    if key in ("sell", "strong_sell"):
        return label, "red"
    if key in ("hold", "underperform"):
        return label, "yellow"
    return label, "white"


def extract_market_sentiment(
    ticker: yf.Ticker,
    current_price: float,
    history,
) -> MarketSentiment:
    """Pull analyst consensus, target price, and 50-day MA technical trend."""
    info: dict[str, Any] = ticker.info or {}

    recommendation_key = info.get("recommendationKey")
    if isinstance(recommendation_key, str):
        recommendation_key = recommendation_key.strip() or None
    else:
        recommendation_key = None

    recommendation_label, recommendation_style = _format_recommendation(recommendation_key)
    analyst_count = _safe_int(info.get("numberOfAnalystOpinions"))

    if analyst_count is not None and recommendation_label != "N/A":
        recommendation_label = f"{recommendation_label} ({analyst_count} analysts)"
    elif analyst_count is not None:
        recommendation_label = f"N/A ({analyst_count} analysts)"

    target_mean_price = _safe_float(info.get("targetMeanPrice"))
    target_vs_current_pct: Optional[float] = None
    target_label = "N/A"
    target_style = "dim"

    if target_mean_price is not None and current_price > 0:
        target_vs_current_pct = ((target_mean_price - current_price) / current_price) * 100.0
        target_label = (
            f"${target_mean_price:,.2f} ({target_vs_current_pct:+.1f}% vs current)"
        )
        if target_vs_current_pct > 0:
            target_style = "green"
        elif target_vs_current_pct < 0:
            target_style = "red"
        else:
            target_style = "yellow"
    elif target_mean_price is not None:
        target_label = f"${target_mean_price:,.2f}"

    sma_50: Optional[float] = None
    technical_label = "N/A"
    technical_style = "dim"
    technical_emoji = "—"
    deep_oversold_threshold = 15.0  # percent below 50-day MA

    if history is not None and not history.empty and "Close" in history.columns:
        closes = history["Close"].dropna()
        if len(closes) >= 50:
            sma_50 = float(closes.tail(50).mean())
        elif len(closes) >= 2:
            sma_50 = float(closes.mean())

        if sma_50 is not None and current_price > 0:
            distance_pct = ((current_price - sma_50) / sma_50) * 100.0
            if current_price < sma_50:
                technical_label = (
                    f"BEARISH / SELL (price {distance_pct:+.1f}% vs 50-day MA "
                    f"${sma_50:,.2f})"
                )
                technical_style = "red"
                technical_emoji = "🔴"
                if distance_pct <= -deep_oversold_threshold:
                    technical_label += " — deeply oversold, potential bounce zone"
            else:
                technical_label = (
                    f"BULLISH / BUY (price {distance_pct:+.1f}% vs 50-day MA "
                    f"${sma_50:,.2f})"
                )
                technical_style = "green"
                technical_emoji = "🟢"

    volume_control_label, volume_control_style = analyze_volume_control(history)

    return MarketSentiment(
        recommendation_key=recommendation_key,
        recommendation_label=recommendation_label,
        recommendation_style=recommendation_style,
        analyst_count=analyst_count,
        target_mean_price=target_mean_price,
        target_vs_current_pct=target_vs_current_pct,
        target_label=target_label,
        target_style=target_style,
        sma_50=sma_50,
        technical_label=technical_label,
        technical_style=technical_style,
        technical_emoji=technical_emoji,
        volume_control_label=volume_control_label,
        volume_control_style=volume_control_style,
    )


def apply_fundamental_override(
    signal: str,
    verdict: str,
    verdict_detail: str,
    fundamentals: FundamentalHealth,
) -> tuple[str, str, str]:
    """
    If a buy/DCA signal fires but cash runway is red, downgrade to a warning.
    """
    is_buy_path = verdict == "BUY SIGNAL" and signal in (
        "BUY",
        "EXECUTE DOLLAR COST AVERAGE (BUY)",
    )
    if is_buy_path and fundamentals.cash_runway_status == "red":
        return (
            signal,
            "WARNING",
            "[WARNING] Technicals suggest buying/DCA, but the company has a high "
            "cash burn rate with less than 1 year of runway. Exercise extreme caution.",
        )
    return signal, verdict, verdict_detail


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------


class AnalysisResult:
    """Container for all metrics and the final trading verdict."""

    def __init__(
        self,
        symbol: str,
        asset_type: str,
        current_price: float,
        peak_price: Optional[float],
        peak_date: Optional[datetime],
        drop_from_peak_pct: Optional[float],
        initial_price: Optional[float],
        drop_from_initial_pct: Optional[float],
        shares_outstanding: Optional[float],
        accumulated_volume: Optional[float],
        turnover: Optional[float],
        min_turnover: float,
        max_turnover: float,
        signal: str,
        verdict: str,
        verdict_detail: str,
        fundamentals: FundamentalHealth,
        sentiment: MarketSentiment,
    ):
        self.symbol = symbol
        self.asset_type = asset_type
        self.current_price = current_price
        self.peak_price = peak_price
        self.peak_date = peak_date
        self.drop_from_peak_pct = drop_from_peak_pct
        self.initial_price = initial_price
        self.drop_from_initial_pct = drop_from_initial_pct
        self.shares_outstanding = shares_outstanding
        self.accumulated_volume = accumulated_volume
        self.turnover = turnover
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover
        self.signal = signal
        self.verdict = verdict
        self.verdict_detail = verdict_detail
        self.fundamentals = fundamentals
        self.sentiment = sentiment


def analyze(
    symbol: str,
    asset_type: str,
    initial_price: Optional[float] = None,
) -> AnalysisResult:
    """Run the full falling-knife analysis for a single ticker."""
    min_turnover, max_turnover = TURNOVER_THRESHOLDS[asset_type]

    ticker = fetch_ticker_data(symbol)
    history = ticker.history(period=HISTORY_PERIOD)

    if history.empty:
        raise ValueError(
            f"No historical data returned for '{symbol}'. "
            "Verify the ticker symbol and try again."
        )

    current_price = get_current_price(ticker, history)
    if current_price is None:
        raise ValueError(f"Could not determine current price for '{symbol}'.")

    fundamentals = extract_fundamental_health(ticker)
    sentiment = extract_market_sentiment(ticker, current_price, history)

    # --- Path 1: existing position with >= 20% drop → DCA buy ---
    drop_from_initial_pct: Optional[float] = None
    signal: str
    verdict: str
    verdict_detail: str
    peak_price: Optional[float] = None
    peak_date: Optional[datetime] = None
    drop_from_peak_pct: Optional[float] = None
    shares_outstanding: Optional[float] = None
    accumulated_volume: Optional[float] = None
    turnover: Optional[float] = None

    if initial_price is not None and initial_price > 0:
        drop_from_initial_pct = pct_drop(initial_price, current_price)
        if drop_from_initial_pct >= INITIAL_DROP_THRESHOLD:
            signal = "EXECUTE DOLLAR COST AVERAGE (BUY)"
            verdict = "BUY SIGNAL"
            verdict_detail = (
                f"Position is down {drop_from_initial_pct:.1f}% from your entry "
                f"(${initial_price:,.2f}). Threshold met — execute DCA buy."
            )
            signal, verdict, verdict_detail = apply_fundamental_override(
                signal, verdict, verdict_detail, fundamentals
            )
            return AnalysisResult(
                symbol=symbol.upper(),
                asset_type=asset_type,
                current_price=current_price,
                peak_price=peak_price,
                peak_date=peak_date,
                drop_from_peak_pct=drop_from_peak_pct,
                initial_price=initial_price,
                drop_from_initial_pct=drop_from_initial_pct,
                shares_outstanding=shares_outstanding,
                accumulated_volume=accumulated_volume,
                turnover=turnover,
                min_turnover=min_turnover,
                max_turnover=max_turnover,
                signal=signal,
                verdict=verdict,
                verdict_detail=verdict_detail,
                fundamentals=fundamentals,
                sentiment=sentiment,
            )

    # --- Path 2: volume / turnover check ---
    peak_price, peak_date = find_recent_peak(history)
    if peak_price is None or peak_date is None:
        raise ValueError(f"Could not determine recent peak for '{symbol}'.")

    drop_from_peak_pct = pct_drop(peak_price, current_price)
    accumulated_volume = accumulated_volume_since(history, peak_date)
    shares_outstanding = get_shares_outstanding(ticker)

    turnover: Optional[float] = None
    if (
        accumulated_volume is not None
        and shares_outstanding is not None
        and shares_outstanding > 0
    ):
        turnover = accumulated_volume / shares_outstanding

    if turnover is None:
        signal = "HOLD/WAIT"
        verdict = "HOLD/WAIT"
        verdict_detail = (
            "Turnover could not be calculated (missing shares outstanding or volume data). "
            "Do not catch the falling knife yet."
        )
    elif turnover >= min_turnover:
        signal = "BUY"
        verdict = "BUY SIGNAL"
        verdict_detail = (
            "Volume confirmed — shareholder base has shifted, safe to execute buy."
        )
    else:
        signal = "HOLD/WAIT"
        verdict = "HOLD/WAIT"
        verdict_detail = (
            "Turnover is still too low. Do not catch the falling knife yet."
        )

    signal, verdict, verdict_detail = apply_fundamental_override(
        signal, verdict, verdict_detail, fundamentals
    )

    return AnalysisResult(
        symbol=symbol.upper(),
        asset_type=asset_type,
        current_price=current_price,
        peak_price=peak_price,
        peak_date=peak_date,
        drop_from_peak_pct=drop_from_peak_pct,
        initial_price=initial_price,
        drop_from_initial_pct=drop_from_initial_pct,
        shares_outstanding=shares_outstanding,
        accumulated_volume=accumulated_volume,
        turnover=turnover,
        min_turnover=min_turnover,
        max_turnover=max_turnover,
        signal=signal,
        verdict=verdict,
        verdict_detail=verdict_detail,
        fundamentals=fundamentals,
        sentiment=sentiment,
    )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------


def _fmt_large_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{sign}{magnitude / 1_000_000_000:.2f} B"
    if magnitude >= 1_000_000:
        return f"{sign}{magnitude / 1_000_000:.2f} M"
    if magnitude >= 1_000:
        return f"{sign}{magnitude / 1_000:.2f} K"
    return f"{sign}{magnitude:,.0f}"


def print_dashboard(result: AnalysisResult) -> None:
    """Render the summary dashboard and final verdict."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]Falling Knife Trading Agent[/bold cyan]\n"
            "[dim]Michael Burry-inspired volume turnover strategy[/dim]",
            box=box.DOUBLE,
            border_style="cyan",
        )
    )

    # --- Price summary ---
    price_table = Table(title="Price Summary", box=box.ROUNDED, show_header=True)
    price_table.add_column("Metric", style="bold")
    price_table.add_column("Value", justify="right")

    price_table.add_row("Ticker", result.symbol)
    price_table.add_row("Asset Type", result.asset_type)
    price_table.add_row("Current Price", f"${result.current_price:,.2f}")

    if result.peak_price is not None:
        peak_date_str = (
            result.peak_date.strftime("%Y-%m-%d") if result.peak_date else "N/A"
        )
        price_table.add_row("Recent Peak (12M)", f"${result.peak_price:,.2f}")
        price_table.add_row("Peak Date", peak_date_str)
        if result.drop_from_peak_pct is not None:
            drop_style = "red" if result.drop_from_peak_pct > 0 else "green"
            price_table.add_row(
                "Drop from Peak",
                Text(f"{result.drop_from_peak_pct:.2f}%", style=drop_style),
            )

    if result.initial_price is not None:
        price_table.add_row("Your Entry Price", f"${result.initial_price:,.2f}")
        if result.drop_from_initial_pct is not None:
            drop_style = "red" if result.drop_from_initial_pct > 0 else "green"
            price_table.add_row(
                "Drop from Entry",
                Text(f"{result.drop_from_initial_pct:.2f}%", style=drop_style),
            )

    console.print(price_table)

    # --- Volume / turnover (only for turnover path) ---
    if result.peak_price is not None:
        vol_table = Table(
            title="Volume & Turnover Analysis", box=box.ROUNDED, show_header=True
        )
        vol_table.add_column("Metric", style="bold")
        vol_table.add_column("Value", justify="right")

        vol_table.add_row(
            "Shares Outstanding", _fmt_large_number(result.shares_outstanding)
        )
        vol_table.add_row(
            "Accumulated Volume (since peak)", _fmt_large_number(result.accumulated_volume)
        )

        if result.turnover is not None:
            turnover_str = f"{result.turnover:.2f}x"
            meets = result.turnover >= result.min_turnover
            turnover_style = "green" if meets else "yellow"
            vol_table.add_row(
                "Calculated Turnover",
                Text(turnover_str, style=turnover_style),
            )
        else:
            vol_table.add_row("Calculated Turnover", "N/A")

        vol_table.add_row(
            "Required Turnover (min)",
            f"{result.min_turnover:.1f}x",
        )
        vol_table.add_row(
            "Required Turnover (target)",
            f"{result.max_turnover:.1f}x",
        )

        console.print(vol_table)

    # --- Fundamental health ---
    fund = result.fundamentals
    fund_table = Table(
        title="Fundamental Health Check", box=box.ROUNDED, show_header=True
    )
    fund_table.add_column("Metric", style="bold")
    fund_table.add_column("Value", justify="right")
    fund_table.add_column("Status", justify="center")

    fund_table.add_row("Total Cash", _fmt_large_number(fund.total_cash), "—")
    fund_table.add_row("Total Debt", _fmt_large_number(fund.total_debt), "—")

    if fund.debt_to_cash_ratio is not None:
        debt_cash_value = f"{fund.debt_to_cash_ratio:.2f}x"
    else:
        debt_cash_value = "N/A"
    fund_table.add_row(
        "Debt-to-Cash Ratio",
        debt_cash_value,
        _status_emoji(fund.debt_to_cash_status),
    )

    fund_table.add_row(
        "Operating Cash Flow", _fmt_large_number(fund.operating_cashflow), "—"
    )

    if fund.cash_runway_years is not None:
        runway_value = f"{fund.cash_runway_years:.1f} years"
    else:
        runway_value = fund.cash_runway_note
    fund_table.add_row(
        "Cash Runway",
        runway_value,
        _status_emoji(fund.cash_runway_status),
    )

    ratio_value = f"{fund.current_ratio:.2f}" if fund.current_ratio is not None else "N/A"
    fund_table.add_row(
        "Current Ratio",
        ratio_value,
        _status_emoji(fund.current_ratio_status),
    )

    trailing_pe_style = "dim" if fund.trailing_pe is None else "white"
    forward_pe_style = "dim" if fund.forward_pe is None else "white"
    fund_table.add_row(
        "Trailing P/E",
        Text(fund.trailing_pe_label, style=trailing_pe_style),
        "—",
    )
    fund_table.add_row(
        "Forward P/E",
        Text(fund.forward_pe_label, style=forward_pe_style),
        "—",
    )

    revenue_growth_style = "bold yellow" if fund.revenue_growth_emphasized else "white"
    if fund.revenue_growth_pct is not None and not fund.revenue_growth_emphasized:
        if fund.revenue_growth_pct > 5:
            revenue_growth_style = "green"
        elif fund.revenue_growth_pct < -5:
            revenue_growth_style = "red"
    fund_table.add_row(
        "Revenue Growth",
        Text(fund.revenue_growth_label, style=revenue_growth_style),
        _status_emoji(
            "green"
            if fund.revenue_growth_pct is not None and fund.revenue_growth_pct > 5
            else "red"
            if fund.revenue_growth_pct is not None and fund.revenue_growth_pct < -5
            else "neutral"
            if fund.revenue_growth_pct is not None
            else "unknown"
        ),
    )

    fund_table.add_row(
        "Revenue / EPS Trend",
        fund.trend_label,
        _status_emoji(fund.trend_status),
    )

    console.print(fund_table)

    # --- Market sentiment ---
    sent = result.sentiment
    sentiment_table = Table(
        title="Market Sentiment Summary", box=box.ROUNDED, show_header=True
    )
    sentiment_table.add_column("Metric", style="bold")
    sentiment_table.add_column("Value")

    sentiment_table.add_row(
        "Analyst Recommendation",
        Text(sent.recommendation_label, style=sent.recommendation_style),
    )
    sentiment_table.add_row(
        "Analyst Target Price",
        Text(sent.target_label, style=sent.target_style),
    )
    sentiment_table.add_row(
        "Technical Summary (50-day MA)",
        Text(
            f"{sent.technical_emoji} {sent.technical_label}",
            style=sent.technical_style,
        ),
    )
    sentiment_table.add_row(
        "Volume Control (Past 5 Days)",
        Text(sent.volume_control_label, style=sent.volume_control_style),
    )

    console.print(sentiment_table)

    # --- Verdict ---
    is_warning = result.verdict == "WARNING"
    is_buy = result.verdict == "BUY SIGNAL"
    if is_warning:
        border = "red"
        icon = "⚠"
    elif is_buy:
        border = "green"
        icon = "✓"
    else:
        border = "yellow"
        icon = "⏸"

    if is_warning:
        headline = result.verdict_detail
    elif result.signal == "EXECUTE DOLLAR COST AVERAGE (BUY)":
        headline = f"[{result.verdict}] EXECUTE DOLLAR COST AVERAGE (BUY)"
    elif is_buy:
        headline = (
            f"[{result.verdict}] Volume confirmed — "
            "Shareholder base has shifted, safe to execute buy"
        )
    else:
        headline = (
            f"[{result.verdict}] Turnover is still too low. "
            "Do not catch the falling knife yet."
        )

    detail = result.verdict_detail if not is_warning else (
        f"Underlying signal: {result.signal}. Review fundamental health before acting."
    )

    console.print()
    console.print(
        Panel(
            f"[bold {border}]{icon} {headline}[/bold {border}]\n\n"
            f"[dim]{detail}[/dim]",
            title="Final Verdict",
            border_style=border,
            box=box.HEAVY,
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knife_agent",
        description=(
            "Falling Knife Trading Agent — scale into dropping assets using "
            "volume turnover to avoid catching a falling knife prematurely."
        ),
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock or ETF symbol (e.g. PLTR, SPY, QQQ).",
    )
    parser.add_argument(
        "--initial_price",
        type=float,
        default=None,
        help="Price at which you previously bought the asset (optional).",
    )
    parser.add_argument(
        "--type",
        choices=ASSET_TYPES,
        default="Standard",
        help="Asset classification for turnover thresholds (default: Standard).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.initial_price is not None and args.initial_price <= 0:
        console.print("[red]Error:[/red] --initial_price must be a positive number.")
        return 1

    try:
        result = analyze(
            symbol=args.ticker,
            asset_type=args.type,
            initial_price=args.initial_price,
        )
        print_dashboard(result)
        return 0 if result.verdict == "BUY SIGNAL" else 2
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
