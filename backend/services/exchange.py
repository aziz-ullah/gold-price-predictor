"""Utilities for retrieving live exchange rates with safe fallbacks."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import requests


DEFAULT_RATE = 280.0
EXCHANGE_API_URL = "https://open.er-api.com/v6/latest/USD"


def _env_rate() -> Optional[float]:
    value = os.getenv("USD_TO_PKR")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _fetch_live_rate() -> Optional[float]:
    try:
        response = requests.get(EXCHANGE_API_URL, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None

    rates = payload.get("rates") or {}
    rate = rates.get("PKR")
    if isinstance(rate, (int, float)) and rate > 0:
        return float(rate)
    return None


def get_usd_to_pkr_rate() -> float:
    env_rate = _env_rate()
    if env_rate and env_rate > 0:
        return env_rate

    live_rate = _fetch_live_rate()
    if live_rate and live_rate > 0:
        return round(live_rate, 4)

    return DEFAULT_RATE
