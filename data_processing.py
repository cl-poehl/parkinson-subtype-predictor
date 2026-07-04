"""Data processing utilities for longitudinal patient measurements.

This module provides functions to filter sparse time points and interpolate
longitudinal scores onto a fixed time grid per patient.
"""

import numpy as np
import pandas as pd


def filter_sparse_timepoints(
    df: pd.DataFrame,
    score_col: str,
    time_col: str = "Disease_duration",
    subtype_col: str = "Subtype",
    minimum_support: int = 10,
) -> pd.DataFrame:
    """Return df rows whose time points meet the minimum per-subtype support threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table that includes subtype identifiers and longitudinal measurements.
    score_col : str
        Column name holding the score to summarize.
    time_col : str, default "Disease_duration"
        Column describing the longitudinal axis.
    subtype_col : str, default "Subtype"
        Categorical variable describing patient subtypes.
    minimum_support : int, default 10
        Minimum number of non-missing scores required per subtype at a time point
        for it to be retained.

    Returns
    -------
    pandas.DataFrame
        Subset of the input table with only rows whose time points meet the
        minimum support requirement for every subtype.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()
    subtype_values = df[subtype_col].dropna().unique()

    for subtype in subtype_values:
        subtype_data = filtered[filtered[subtype_col] == subtype]
        if subtype_data.empty:
            continue

        support_counts = subtype_data.groupby(time_col)[score_col].count()
        supported_times = support_counts[support_counts >= minimum_support].index
        filtered = filtered[filtered[time_col].isin(supported_times)]

        if filtered.empty:
            break

    return filtered


def interpolate_scores_to_grid(
    df: pd.DataFrame,
    score_col: str,
    freq: float = 1,
    time_col: str = "Disease_duration",
    patno_col: str = "PATNO",
    additional_cols: list[str] = None,
    min_time: float = 0,
    max_time: float = None,
    method: str = "linear",
) -> pd.DataFrame:
    """Return a DataFrame interpolated onto a fixed time grid per patient.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing per-patient longitudinal measurements.
    score_col : str
        Column with the measurement values to interpolate.
    time_col : str
        Column containing the time axis (numeric and ordered within each patient).
    patno_col : str
        Patient identifier column used to group measurements.
    freq : float, default 1
        Spacing between consecutive time grid points (same units as `time_col`).
    additional_cols : list of str, optional
        Provide additional columns from the input DataFrame to be carried over. Defaults to ["Subtype"].
    min_time : float, optional
        Explicit start of the interpolation grid; defaults to 0
    max_time : float, optional
        Explicit end of the interpolation grid; defaults to the maximum observed time.
    method : str, default "linear"
        Interpolation scheme forwarded to `pandas.Series.interpolate`.

    Returns
    -------
    pandas.DataFrame
        Table with interpolated scores for every patient/time grid combination that
        lies between their first and last observation. Time points represented by
        fewer than 10 scores across patients are removed.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame or if parameters are invalid.
    """
    # Parameter checks.
    required_cols = {score_col, time_col, patno_col}
    missing_cols = sorted(required_cols.difference(df.columns))
    if additional_cols is None:
        additional_cols = ["Subtype"]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
    if freq <= 0:
        raise ValueError("freq must be positive.")
    if additional_cols is None:
        additional_cols = []
    if df.empty:
        return df[[patno_col, time_col, score_col]].copy()

    # Work on a clean copy with the required columns and a stable ordering per patient.
    working = (
        df[[patno_col, time_col, score_col] + additional_cols]
        .dropna(subset=[patno_col, time_col, score_col])
        .sort_values([patno_col, time_col])
    )

    # further parameter checks
    if working.empty:
        return working
    if min_time is None:
        min_time = working[time_col].min()
    if max_time is None:
        max_time = working[time_col].max()
    if min_time > max_time:
        raise ValueError("min_time must be <= max_time.")

    # Construct the common time grid
    epsilon = freq * 0.5
    time_grid = np.arange(min_time, max_time + epsilon, freq)
    result_frames = []

    # Interpolate per patient
    for patno, group in working.groupby(patno_col, sort=False):
        # Ensure a monotonic signal per patient before interpolation.
        series = (
            group.drop_duplicates(subset=time_col, keep="last")
            .set_index(time_col)[score_col]
            .sort_index()
        )
        if series.empty:
            continue

        # Only interpolate within observed time bounds to avoid extrapolation artifacts.
        patient_grid = time_grid[
            (time_grid >= series.index.min()) & (time_grid <= series.index.max())
        ]
        if patient_grid.size == 0:
            continue

        # Reindex to the combined index of observed and grid points
        combined_index = np.union1d(series.index.to_numpy(), patient_grid)

        # Perform the interpolation
        interpolated = (
            series.reindex(combined_index)
            .sort_index()
            .interpolate(method=method, limit_direction="both")
        )

        # Extract only the grid points that lie within the patient's observed time range
        interpolated = interpolated.loc[patient_grid]

        # convert to a dataframe, make index a column again
        interpolated.index.name = time_col
        frame = interpolated.reset_index()

        # add patient id and additional columns
        frame[patno_col] = patno
        for col in additional_cols:
            if col in df.columns:
                value = group.iloc[0][col]
                frame[col] = value
        result_frames.append(frame[[patno_col, time_col, score_col] + additional_cols])

    # If no patient had any valid data, return an empty frame with the right columns
    if not result_frames:
        return working.iloc[0:0]

    # Concatenate all patient frames and sort
    result = pd.concat(result_frames, ignore_index=True)
    return result.sort_values([patno_col, time_col] + additional_cols).reset_index(
        drop=True
    )
