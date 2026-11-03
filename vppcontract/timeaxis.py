"""과제 시간축, 평가 기간, 공휴일 달력.

전역 시간축의 h=0은 2021-01-01 01:00 KST이고, 시각은 우측닫힘 1시간 구간의 끝이다.
대상일은 date(ts−1h)이므로 대상일 index는 h//24, slot은 h%24(slot 0 = 01시, slot 23 = 다음날 00시)이다.
각 기간의 cutoff는 첫 대상일의 예측시점(전일 17시)이다.
robust fold test의 fold(FOLD_PERIODS)는 V·H와 같은 규칙으로 cutoff만 3개월씩 다른 12개월 평가 기간이고, 계절 조건은 SEASONS를 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
T0 = pd.Timestamp("2021-01-01 01:00", tz=TZ)
HOUR_NS = 3_600_000_000_000
N_DAYS = 1_977
N_HOURS = N_DAYS * 24
N_ACTUAL = 38_688  # generation·observed가 있는 2025-06-01 00:00까지
K_PEAK_HOURS = (11, 12, 13, 14, 15)
GROUPS = ("G1", "G2")
MINIMUM_GROUP_CAPACITY_KW = 20_000.0


def hour_index(ts) -> tuple[np.ndarray, np.ndarray]:
    """timestamp를 h로 바꾼다. 단위(ms·us·ns)와 무관하며, 정시 여부를 함께 돌려준다."""
    values = pd.DatetimeIndex(ts)
    if values.tz is None:
        raise ValueError("timestamp에는 timezone이 있어야 합니다.")
    delta = values.as_unit("ns").asi8 - T0.value
    return np.floor_divide(delta, HOUR_NS), delta % HOUR_NS == 0


def to_h(timestamp) -> int:
    h, on_hour = hour_index([pd.Timestamp(timestamp)])
    if not on_hour[0]:
        raise ValueError(f"정시가 아닌 시각입니다: {timestamp}")
    return int(h[0])


def timestamps(h) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(T0 + pd.to_timedelta(np.asarray(h, dtype=np.int64), unit="h"))


def clock_hour(h) -> np.ndarray:
    return (np.asarray(h) + 1) % 24


def k_weight(h) -> np.ndarray:
    """정산 시간 가중 k. 표시 시각의 시가 11~15이면 3, 그 밖은 1이다."""
    return np.where(np.isin(clock_hour(h), K_PEAK_HOURS), 3, 1).astype(np.int8)


def day_of(h):
    return np.asarray(h) // 24


def delivery_dates(days) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(T0.normalize().tz_localize(None) + pd.to_timedelta(np.asarray(days, dtype=np.int64), unit="D"))


def delivery_month(h) -> np.ndarray:
    return delivery_dates(day_of(h)).strftime("%Y-%m").to_numpy()


def calendar_month(days) -> np.ndarray:
    """대상일의 달력 월(1~12)."""
    return delivery_dates(days).month.to_numpy()


def day_of_year(days) -> np.ndarray:
    return delivery_dates(days).dayofyear.to_numpy()


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str
    cutoff: str

    @property
    def start_h(self) -> int:
        return to_h(self.start)

    @property
    def end_h(self) -> int:
        return to_h(self.end)

    @property
    def cutoff_h(self) -> int:
        return to_h(self.cutoff)

    @property
    def hours(self) -> np.ndarray:
        return np.arange(self.start_h, self.end_h + 1)

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.start_h // 24, self.end_h // 24 + 1)

    @property
    def has_labels(self) -> bool:
        return self.end_h < N_ACTUAL


PERIODS = {
    "D": Period("D", "2022-06-01 01:00:00+09:00", "2023-06-01 00:00:00+09:00", "2022-05-31 17:00:00+09:00"),
    "V": Period("V", "2023-06-01 01:00:00+09:00", "2024-06-01 00:00:00+09:00", "2023-05-31 17:00:00+09:00"),
    "fold-2309": Period("fold-2309", "2023-09-01 01:00:00+09:00", "2024-09-01 00:00:00+09:00", "2023-08-31 17:00:00+09:00"),
    "fold-2312": Period("fold-2312", "2023-12-01 01:00:00+09:00", "2024-12-01 00:00:00+09:00", "2023-11-30 17:00:00+09:00"),
    "fold-2403": Period("fold-2403", "2024-03-01 01:00:00+09:00", "2025-03-01 00:00:00+09:00", "2024-02-29 17:00:00+09:00"),
    "H": Period("H", "2024-06-01 01:00:00+09:00", "2025-06-01 00:00:00+09:00", "2024-05-31 17:00:00+09:00"),
    "B": Period("B", "2024-12-25 01:00:00+09:00", "2025-06-01 00:00:00+09:00", "2024-12-24 17:00:00+09:00"),
    "test": Period("test", "2025-06-01 01:00:00+09:00", "2026-06-01 00:00:00+09:00", "2025-05-31 17:00:00+09:00"),
}


def period(name: str) -> Period:
    if name not in PERIODS:
        raise ValueError(f"알 수 없는 기간입니다: {name} (기간: {', '.join(PERIODS)})")
    return PERIODS[name]


FOLD_PERIODS = ("V", "fold-2309", "fold-2312", "fold-2403", "H")
SEASONS = {"봄": (3, 4, 5), "여름": (6, 7, 8), "가을": (9, 10, 11), "겨울": (12, 1, 2)}


def season_index(days) -> np.ndarray:
    """대상일의 계절 index(SEASONS 순서: 봄 0·여름 1·가을 2·겨울 3). 계절은 대상일의 월로 정한다."""
    month = delivery_dates(days).month.to_numpy()
    return np.select([np.isin(month, m) for m in SEASONS.values()], range(len(SEASONS)))


SEASON_IDS = ("spring", "summer", "autumn", "winter")     # SEASONS 순서의 영문 이름. 설정·명령행·산출물 이름에 쓴다.


def season_mask(days, season: str, margin_months: int = 0) -> np.ndarray:
    """대상일 가운데 계절 season(SEASON_IDS의 이름)에 속하는 날. margin_months > 0이면 계절의 앞뒤로 그 개월 수만큼 넓힌 월까지 포함한다
    (예: 겨울 12~2월, margin 1 → 11~3월)."""
    if season not in SEASON_IDS:
        raise ValueError(f"계절은 {SEASON_IDS} 중 하나입니다: {season}")
    if int(margin_months) < 0:
        raise ValueError(f"margin_months는 0 이상입니다: {margin_months}")
    if not margin_months:
        return season_index(days) == SEASON_IDS.index(season)
    months = list(SEASONS.values())[SEASON_IDS.index(season)]
    wide = set(months) | {(months[0] - 2 - k) % 12 + 1 for k in range(int(margin_months))} \
        | {(months[-1] + k) % 12 + 1 for k in range(int(margin_months))}
    return np.isin(delivery_dates(days).month.to_numpy(), sorted(wide))


class HolidayCalendar:
    """공휴일 목록(config)으로 대상일이 쉬는 날(토·일·공휴일·근로자의 날)인지 정한다.

    대상일 d의 예측시점은 d−1일 17시이다. 공고일(announced)이 있는 공휴일은 예측시점이 속한 날의 전날까지 공고된 것만 공휴일로 본다.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        month, day = (int(v) for v in self.config["labor_day"].split("-"))
        self.labor = (month, day)
        self.announced = {date.fromisoformat(item["date"]): (date.fromisoformat(item["announced"]) if "announced" in item else None)
                          for item in self.config["holidays"]}

    @property
    def rules(self) -> dict:
        """판정에 쓰는 규칙(근로자의 날, 공휴일과 공고일). 설명·출처·이름 문구는 넣지 않는다."""
        return {"labor_day": list(self.labor),
                "holidays": [{"date": day.isoformat(), "announced": announced.isoformat() if announced else None}
                             for day, announced in sorted(self.announced.items())]}

    def is_holiday(self, d: date, forecast_date: date) -> bool:
        """forecast_date(예측시점이 속한 날) 기준으로 알려진 공휴일인가."""
        if d not in self.announced:
            return False
        announced = self.announced[d]
        return announced is None or announced <= forecast_date - timedelta(days=1)

    def is_labor_day(self, d: date) -> bool:
        return (d.month, d.day) == self.labor

    def is_off(self, d: date, forecast_date: date) -> bool:
        """토·일·공휴일·근로자의 날."""
        return d.weekday() >= 5 or self.is_holiday(d, forecast_date) or self.is_labor_day(d)

    def day_kinds(self, days) -> tuple[np.ndarray, np.ndarray]:
        """대상일 index 배열의 (토요일, 일요일·공휴일·근로자의 날) 여부. 공휴일인 토요일은 뒤쪽에 넣는다. 예측시점이 속한 날은 대상일 전날이다."""
        dates = [d.date() for d in delivery_dates(days)]
        rest = np.array([d.weekday() == 6 or self.is_holiday(d, d - timedelta(days=1)) or self.is_labor_day(d) for d in dates], dtype=bool)
        saturday = np.array([d.weekday() == 5 for d in dates], dtype=bool) & ~rest
        return saturday, rest
