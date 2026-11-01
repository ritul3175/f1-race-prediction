import os
from typing import Dict, List

import fastf1
import pandas as pd


def _init_cache():
    """Ensure cache directory exists and enable FastF1 cache."""
    cache_dir = os.path.join(os.path.dirname(__file__), "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def get_event_schedule(year: int) -> List[str]:
    """
    Get list of Grand Prix event names for a given season.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.

    Returns
    -------
    List[str]
        List of event names (e.g. ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', ...])
    """
    _init_cache()
    try:
        schedule = fastf1.get_event_schedule(year)
    except ValueError as e:
        raise ValueError(
            f"Could not load F1 schedule for {year}. "
            "Check your internet connection. If you're online, the FastF1/Ergast API may be "
            "temporarily unavailable or the year may not have schedule data yet."
        ) from e
    # Filter out testing events
    races = schedule[schedule["EventFormat"] != "testing"]
    return races["EventName"].tolist()


def get_driver_info(year: int, venue: str) -> Dict[str, str]:
    """
    Get a mapping of driver abbreviations to full names.

    Parameters
    ----------
    year : int
        Season year.
    venue : str
        Grand Prix name.

    Returns
    -------
    Dict[str, str]
        Mapping like {'VER': 'Max Verstappen', 'HAM': 'Lewis Hamilton', ...}
    """
    _init_cache()
    try:
        session = fastf1.get_session(year, venue, "FP1")
        session.load()
        drivers = session.drivers
        driver_map = {}
        for drv in drivers:
            info = session.get_driver(drv)
            abbrev = info.get("Abbreviation", drv)
            full_name = info.get("FullName", abbrev)
            driver_map[abbrev] = full_name
        return driver_map
    except Exception:
        return {}


def get_practice_laps(year: int, venue: str) -> pd.DataFrame:
    """
    Download and return all practice session laps for the selected
    year and venue using the FastF1 API.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.
    venue : str
        Grand Prix name, e.g. 'Bahrain', 'Monza'.

    Returns
    -------
    pandas.DataFrame
        Combined laps from FP1, FP2 and FP3 (if available).
    """
    _init_cache()

    sessions = []
    failures = []
    for session_name in ["FP1", "FP2", "FP3"]:
        try:
            session = fastf1.get_session(year, venue, session_name)
            session.load()
            laps = session.laps
            if not laps.empty:
                laps = laps.copy()
                laps["Session"] = session_name
                sessions.append(laps)
        except Exception as e:
            failures.append(f"{session_name}: {e}")
            continue

    if not sessions:
        failure_details = f" Session loading errors: {'; '.join(failures)}" if failures else ""
        raise ValueError(
            f"No practice lap data is currently available for {year} {venue}. "
            "FastF1 may not have published the session timing data yet. "
            "Try a completed event with available practice data."
            f"{failure_details}"
        )

    return pd.concat(sessions, ignore_index=True)


def get_race_results(year: int, venue: str) -> pd.DataFrame:
    """
    Get actual race finishing positions for a given event.

    Parameters
    ----------
    year : int
        Season year.
    venue : str
        Grand Prix name (e.g. 'Bahrain Grand Prix').

    Returns
    -------
    pandas.DataFrame
        Columns: Driver (abbreviation), Position, FullName (if available).
    """
    _init_cache()
    try:
        session = fastf1.get_session(year, venue, "R")
        session.load()
        results = session.results
        if results is None or len(results) == 0:
            return pd.DataFrame()
        # results has Abbreviation, Position, etc.
        df = results[["Abbreviation", "Position"]].copy()
        df = df.rename(columns={"Abbreviation": "Driver"})
        df = df.dropna(subset=["Position"])
        df["Position"] = df["Position"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()
