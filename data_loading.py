"""Module for loading and preprocessing PPMI data.

The module provides functionality to read PPMI data from CSV files, merge it with subtype information,
and calculate disease duration for each patient.
"""
import os.path

import pandas as pd

from constants import _file_pd_data, _file_subtypes, _basepath


def load_data(file_pd: str = None, file_subtypes: str = None) -> pd.DataFrame:
    """Load and preprocess the PPMI data.

    Parameters
    ----------
    file_pd : str, optional
        Path to the main PPMI data CSV file. If None, uses the default path.
    file_subtypes : str, optional
        Path to the CSV file containing subtype information. If None, uses the default path.

    Returns
    -------
    pandas.DataFrame
        Merged and preprocessed DataFrame with disease duration calculated.
    """
    if file_pd is None:
        file_pd = os.path.join(_basepath, _file_pd_data)
    if file_subtypes is None:
        file_subtypes = os.path.join(_basepath, _file_subtypes)

    data_pd = pd.read_csv(file_pd, low_memory=False)
    data_subtypes = pd.read_csv(file_subtypes)

    data = data_pd.merge(data_subtypes, on="PATNO", how="inner")

    data["Timepoint"] = data["Timepoint"]

    data["Disease_duration"] = (
                                       data["Age_at_BL"] - data["Age_at_diagnosis"]
                               ) * 12  # to months
    data["Disease_duration"] = data["Disease_duration"] + data["Timepoint"]

    # convert subtype column to int
    data["Subtype"] = pd.to_numeric(data["Subtype"])

    return data
