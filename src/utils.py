import pandas as pd


FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "brake_events",
    "hard_brake_events",
    "rain_intensity",
    "is_night",
    "brake_events_per_10km",
    "hard_brake_ratio",
    "rain_heavy",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature engineering on telemetry data.
    This mirrors the kind of feature patterns we'd use in simulation pipelines:
    normalizing events and creating simple interaction terms.
    """
    df = df.copy()

    # Assume trip length ~50km just to create a rate feature
    df["brake_events_per_10km"] = df["brake_events"] / 5.0

    # Avoid divide-by-zero
    df["hard_brake_ratio"] = df["hard_brake_events"] / df["brake_events"].clip(lower=1)

    # Bucket heavy rain
    df["rain_heavy"] = (df["rain_intensity"] >= 0.6).astype(int)

    return df[FEATURE_COLUMNS]