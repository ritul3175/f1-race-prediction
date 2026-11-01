import pandas as pd


def preprocess_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean laps from FastF1: keep only accurate flying laps (not in/out,
    not deleted, under green), then restrict to useful columns.
    """
    if laps.empty:
        return laps

    df = laps.copy()

    # Keep only accurate laps (excludes in/out laps, SC laps, etc.)
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]
    # Exclude deleted lap times (e.g. track limits)
    if "Deleted" in df.columns:
        df = df[df["Deleted"] != True]

    # Drop laps without a valid laptime
    if "LapTime" not in df.columns:
        return df.reset_index(drop=True)
    df = df[df["LapTime"].notna()]
    if df.empty:
        return df.reset_index(drop=True)

    keep_cols = [
        "Driver",
        "Team",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "Session",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()

    # Convert timedeltas to seconds
    time_cols = [
        c for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"] if c in df.columns
    ]
    for c in time_cols:
        df[c] = df[c].dt.total_seconds()

    return df.reset_index(drop=True)


def aggregate_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per driver: focus on best lap and best sector times
    as pace indicators for qualifying/race prediction.
    """
    if df.empty:
        return df

    group_cols = ["Driver"]
    if "Team" in df.columns:
        group_cols.append("Team")

    agg = {}

    # Best lap and best sectors (key for pace)
    for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if c in df.columns:
            agg[c] = ["min", "mean", "std"]

    # Speed traps: use max (faster = better)
    for c in ["SpeedI1", "SpeedI2", "SpeedFL"]:
        if c in df.columns:
            agg[c] = ["max", "mean"]

    if not agg:
        return df.groupby(group_cols).first().reset_index()

    grouped = df.groupby(group_cols).agg(agg)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped
