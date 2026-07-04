"""Functions to calculate likelihood ratios based on score distributions and slopes

This module provides functions to calculate likelihood ratios for patient scores based on their distributions
and slopes. It includes methods for fitting linear mixed effects models to estimate slope distributions,
and calculating likelihoods and likelihood ratios using different statistical methods.
The module supports k-fold cross-validation for robust likelihood ratio estimation.

"""

import warnings

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm  # plain tqdm vermeidet IProgressWarning in headless notebooks

from constants import SCORE_LABELS, SUBTYPE_FAST, SUBTYPE_SLOW, MODEL_LABELS
from data_processing import interpolate_scores_to_grid, filter_sparse_timepoints


def calc_score_slope_distribution(
    data: pd.DataFrame,
    score_col: str,
    subtype_col: str = "Subtype",
    patno_col: str = "PATNO",
    time_col: str = "Disease_duration",
    solver_order: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate the distribution of slopes for a given score using a linear mixed effects model

     uses fixed+random intercept + slope

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes.
    score_col : str
        The score column for which to calculate the slope distribution.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    time_col : str, default "Disease_duration"
     The column name for time in the DataFrame.
    solver_order : list[str] | None, optional
        List of solvers to try in order for fitting the mixed effects model.
        If None, defaults to ["nm", "lbfgs", "cg", "bfgs", "powell"].

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the slope distribution for each patient with subtype information.

    """
    if solver_order is None:
        solver_order = ["nm", "lbfgs", "cg", "bfgs", "powell"]

    # Filter data for the specified score
    data = data[[score_col, time_col, patno_col, subtype_col]].dropna()

    # make patient ID a categorical variable
    data[patno_col] = data[patno_col].astype("category")

    # DataFrame to store slopes
    slopes = pd.DataFrame()

    # fit a linear mixed effects model with random intercept and slope for subtype separately
    for subtype in data[subtype_col].unique():
        sub_data = data[data[subtype_col] == subtype]

        model = MixedLM.from_formula(
            f"{score_col} ~ {time_col}",
            groups=sub_data[patno_col],
            re_formula=f"1+{time_col}",
            data=sub_data,
        )

        cache_key = (score_col, subtype)

        for solver in solver_order:
            try:
                result = model.fit(method=solver)

                # extract fixed slope
                fixed_slope = result.fe_params[time_col]

                # extract random slopes
                random_slopes = np.array(
                    [x[time_col] for x in result.random_effects.values()]
                )
                break
            except Exception as e:
                warnings.warn(
                    "Solver "
                    + solver
                    + " failed for score "
                    + score_col
                    + " with error: "
                    + str(e),
                    UserWarning,
                )
                if solver == solver_order[-1]:
                    raise RuntimeError(
                        f"All solvers failed for score {score_col} and subtype {subtype}."
                    ) from e

        # combine fixed and random slopes
        combined_slopes = fixed_slope + random_slopes

        # add to slopes DataFrame
        temp_df = pd.DataFrame({score_col: combined_slopes, subtype_col: subtype})
        slopes = pd.concat([slopes, temp_df], ignore_index=True)

    return slopes


def calc_likelihood(
    distribution: np.ndarray,
    new_score: float,
    method: str = "z-score",
    num_min_samples: int = 10,
    enable_warnings: bool = True,
) -> float:
    """Calculate the likelihood of a new score given a distribution.

    Parameters
    ----------
    distribution: array-like
        The distribution the score in the reference cohort.
    new_score: float
        The new score to evaluate.
    method: str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    num_min_samples: int, default 10
        Minimum number of samples required in the distribution to calculate likelihood.
    enable_warnings: bool, default True
        Whether to print warnings during calculation if NAN is returned.

    Returns
    -------
    float
        The likelihood of the new score.
    """

    if len(distribution) < num_min_samples:
        if enable_warnings:
            warnings.warn(
                "Not enough samples in distribution to calculate likelihood. Returning NaN.",
                UserWarning,
            )
        return np.nan

    if np.min(distribution) == np.max(distribution):
        if enable_warnings:
            warnings.warn("Distribution has zero variance. Returning NaN.", UserWarning)
        return np.nan

    if method == "empirical_cdf":
        # Empirical CDF method
        percentile = stats.percentileofscore(distribution, new_score, kind="rank") / 100
        likelihood = min(percentile, 1 - percentile) * 2  # Two-tailed

        # ensure likelihood is not zero/one
        if likelihood <= 0:
            likelihood = 1 / (len(distribution) + 1)

        if likelihood >= 1:
            likelihood = 1 - (1 / (len(distribution) + 1))

        return likelihood
    elif method == "z-score":
        # Z-score method
        mean = np.mean(distribution)
        std = np.std(distribution)
        if std == 0:
            if enable_warnings:
                warnings.warn("Standard deviation is zero. Returning NaN.", UserWarning)
            return np.nan
        z_score = (new_score - mean) / std
        likelihood = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
        return likelihood
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods are 'empirical_cdf' and 'z-score'."
        )


def calc_likelihood_ratio(
    df: pd.DataFrame,
    new_score: float,
    score_col: str,
    subtype_col: str = "Subtype",
    method: str = "z-score",
    num_min_samples: int = 10,
    enable_warnings: bool = True,
) -> float:
    """Calculate the likelihood ratio of a new score between two subtypes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table that includes subtype identifiers and score measurements.
    new_score : float
        The new score to evaluate.
    score_col : str
        Column name holding the score to evaluate.
    subtype_col : str, default "Subtype"
        Categorical variable describing patient subtypes. Must contain "Mild" and "Rapid".
    method : str, default "z-score"
        Method to use for likelihood calculation. Options are "empirical_cdf", and "z-score".
    num_min_samples : int, default 10
        Minimum number of samples required in each subtype to calculate likelihood.
    enable_warnings : bool, default True
        Whether to print warnings during calculation if NAN is returned.

    Returns
    -------
    float
        The likelihood ratio of the new score between the two subtypes.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    if df.empty:
        if enable_warnings:
            warnings.warn(
                "Warning: Input DataFrame is empty. Returning NaN.", UserWarning
            )
        return np.nan

    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in DataFrame.")

    if subtype_col not in df.columns:
        raise ValueError(f"Subtype column '{subtype_col}' not found in DataFrame.")

    distribution_1 = df[df[subtype_col] == SUBTYPE_FAST][score_col].dropna().to_numpy()
    distribution_2 = df[df[subtype_col] == SUBTYPE_SLOW][score_col].dropna().to_numpy()

    likelihood_1 = calc_likelihood(
        distribution_1,
        new_score,
        method=method,
        num_min_samples=num_min_samples,
        enable_warnings=enable_warnings,
    )
    likelihood_2 = calc_likelihood(
        distribution_2,
        new_score,
        method=method,
        num_min_samples=num_min_samples,
        enable_warnings=enable_warnings,
    )

    if np.isnan(likelihood_1) or np.isnan(likelihood_2):
        if enable_warnings:
            warnings.warn(
                "Could not calculate likelihood for one of the subtypes. Returning NaN.",
                UserWarning,
            )
        return np.nan

    if likelihood_1 == 0 and likelihood_2 == 0:
        if enable_warnings:
            warnings.warn(
                "Likelihood for both subtypes is zero and "
                + str(likelihood_2)
                + " for subtype 2 with score="
                + str(new_score)
                + ". Subtype 1 distribution: "
                + str(np.mean(distribution_1))
                + " +/- "
                + str(np.std(distribution_1))
                + ", Subtype 2 distribution: "
                + str(np.mean(distribution_2))
                + " +/- "
                + str(np.std(distribution_2))
                + ". Returning NaN.",
                UserWarning,
            )
        return np.nan

    if likelihood_1 == 0:
        if enable_warnings:
            warnings.warn(
                "Likelihood for subtype 1 is zero and "
                + str(likelihood_2)
                + " for subtype 2 with score="
                + str(new_score)
                + ". Subtype 1 distribution: "
                + str(np.mean(distribution_1))
                + " +/- "
                + str(np.std(distribution_1))
                + ", Subtype 2 distribution: "
                + str(np.mean(distribution_2))
                + " +/- "
                + str(np.std(distribution_2))
                + ". Returning small value (0.05).",
                UserWarning,
            )
        return 0.05

    if likelihood_2 == 0:
        if enable_warnings:
            warnings.warn(
                "Likelihood for subtype 2 is zero + and "
                + str(likelihood_1)
                + " for subtype 1 with score="
                + str(new_score)
                + ". Subtype 1 distribution: "
                + str(np.mean(distribution_1))
                + " +/- "
                + str(np.std(distribution_1))
                + ", Subtype 2 distribution: "
                + str(np.mean(distribution_2))
                + " +/- "
                + str(np.std(distribution_2))
                + ". Returning high value (20.0).",
                UserWarning,
            )
        return 20

    likelihood_ratio = likelihood_1 / likelihood_2

    return likelihood_ratio


def filter_missings_by_cutoff(
    data: pd.DataFrame,
    patno_col: str = "PATNO",
    scores: list[str] | None = None,
    cutoff: float = 1,
) -> pd.DataFrame:
    """Returns only patients with missingness below cutoff

    Missingness is calculated as the fraction of missing scores averaged over all timepoints for each patient. Patients
    with a missingness above the specified cutoff are excluded.
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing data from at least one patient and the specified scores with patient IDs.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    scores : list[str], optional
        List of score columns to check for missingness. If None, all scores from SCORE_LABELS are used.
    cutoff : float, default 1
        Maximum allowed fraction of missing scores. Needs to be between 0 and 1. Default is 1 (no filtering)

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with patients below the missingness cutoff.
    """
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    return data.groupby(patno_col).filter(
        lambda x: x[scores].isna().mean().mean() <= cutoff
    )


def introduce_missingness_by_cutoff(
    data: pd.DataFrame,
    patno_col: str = "PATNO",
    scores: list[str] | None = None,
    cutoff: float = 0,
) -> pd.DataFrame:
    """Introduce missingness to patients up to a specified cutoff

    Missingness is introduced by randomly setting score values to NaN until the average missingness per patient
    reaches the specified cutoff.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing data from at least one patient and the specified scores with patient IDs.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    scores : list[str] | None, optional
        List of score columns to introduce missingness to. If None, all scores from SCORE_LABELS are used.
    cutoff : float, default 0
        Target fraction of missing scores per patient. Needs to be between 0 and 1. Default is 0
        (no missingness introduced).

    Returns
    -------
    pandas.DataFrame
        DataFrame with introduced missingness.
    """

    if scores is None:
        scores = list(SCORE_LABELS.keys())

    # do this patient-wise
    i = 0
    for patno, group in data.groupby(patno_col):
        # calculate number of missings already present
        n_missings_expected = int(
            group[scores].shape[0] * group[scores].shape[1] * cutoff
        )

        # if there are not enough missings, introduce more
        n_missings_current = group[scores].isna().sum().sum()
        n_missings_to_introduce = n_missings_expected - n_missings_current

        if n_missings_to_introduce > 0:
            # randomly select combinations of scores/indices to set to NaN (that are not already NaN)
            available_indices = group[scores].isna()
            possible_positions = [
                (i, score)
                for i in group.index
                for score in scores
                if not available_indices.loc[i, score]
            ]
            positions_to_nan = pd.DataFrame(possible_positions).sample(
                n=n_missings_to_introduce, random_state=i
            )
            for _, row in positions_to_nan.iterrows():
                data.loc[row[0], row[1]] = pd.NA
        i += 1

    return data


def filter_follow_up_by_cutoff(
    data: pd.DataFrame,
    patno_col: str = "PATNO",
    time_col: str = "Disease_duration",
    cutoff: float = 0,
) -> pd.DataFrame:
    """Returns only patients with sufficient follow-up.

    Follow-up is calculated last - first Timepoint per patient. Patients with a follow up below the specified cutoff
     are excluded.
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing data from at least one patient and the specified scores with patient IDs and Timepoint.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for timepoints in the DataFrame.
    cutoff : float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Default is 0 (no filtering).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with patients above the follow-up cutoff.
    """

    return data.groupby(patno_col).filter(lambda x: x[time_col].max() - x[time_col].min() >= cutoff)


def shorten_follow_up_to_cutoff(
        data: pd.DataFrame,
        timepoint_col: str = "Disease_duration",
        patno_col: str = "PATNO",
        cutoff: float = np.inf
) -> pd.DataFrame:
    """Shorten follow-up timepoints to a specified cutoff

    Patients are shortened to a given maximum follow up time on patient-level. Follow-up timepoints are calculated
    as last - first Timepoint per patient. Timepoints exceeding the specified cutoff follow up time are removed.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing data from at least one patient and the specified scores with patient IDs and Timepoint.
    timepoint_col : str, default "Disease_duration"
        The column name for timepoints in the DataFrame.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    cutoff : float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed. Default is np.inf
        (no shortening).

    Returns
    -------
    pandas.DataFrame
        DataFrame with shortened follow-up timepoints.
    """

    max_timepoint = data.groupby(patno_col)[timepoint_col].min() + cutoff
    data = data[data.apply(lambda row: row[timepoint_col] <= max_timepoint[row[patno_col]], axis=1)]
    return data


def calc_likelihood_ratios_abs_cv(
    data: pd.DataFrame,
    first_score: bool,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    folds: int = 10,
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate log 10 likelihood ratios for each patient using absolute scores and k-fold cross validation

    Calculates log10 likelihood ratios for each patient based on absolute score values using k-fold cross validation.
    Likelihood ratios are computed for each score at the first available timepoint (or all timepoints if specified),
    and the total log10 likelihood ratio is obtained by summing individual score log10 likelihood ratios across scores.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes.
    first_score : bool
        Whether to use only the first timepoint for each score for likelihood calculation or all available timepoints
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 1
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set in each fold. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set in each fold. Needs to be a float between 0 and 1.
         Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set in each fold. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set in each fold.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    folds : int, default 10
        Number of folds for k-fold cross validation
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from
        SCORE_LABELS if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with log10 likelihood ratios for each score and the total log10 likelihood ratio.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    additional_cols = [subtype_col]
    if scores is None:
        scores = list(SCORE_LABELS.keys())
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)

    # Worker function for interpolating scores to grid in isolated process
    def _interpolate_scores_worker(
        score: str,
        df: pd.DataFrame,
        time_col: str,
        patno_col: str,
        additional_cols: list[str],
        subtype_col: str,
    ) -> tuple[str, pd.DataFrame]:

        data_interpolated_score = interpolate_scores_to_grid(
            df=df,
            score_col=score,
            time_col=time_col,
            patno_col=patno_col,
            additional_cols=additional_cols,
        )

        # filter sparse timepoints
        data_interpolated_score = filter_sparse_timepoints(
            df=data_interpolated_score,
            score_col=score,
            time_col=time_col,
            subtype_col=subtype_col,
        )

        return score, data_interpolated_score

    # Calculate interpolated scores for all scores and patients in parallel
    # Each score is fitted in an isolated subprocess which is destroyed afterwards to avoid memory leaks

    # filter some warning messages that don't affect the results
    warnings.filterwarnings(
        "ignore",
        message=r"A worker stopped while some jobs were given to the executor.*",
        category=UserWarning,
        module=r"joblib\.externals\.loky\.process_executor",
    )

    # parallel execution
    results = Parallel(n_jobs=n_jobs, timeout=9999, backend="loky")(
        delayed(_interpolate_scores_worker)(
            score, data, time_col, patno_col, additional_cols, subtype_col
        )
        for score in tqdm(
            scores,
            disable=not progress_bar,
            desc=f"Submitting score interpolation jobs",
        )
    )

    # Convert results list to dictionary
    data_interpolated = dict(results)

    # prepare likelihood dataframe
    likelihood_ratios = {}

    # k-fold split on patient level
    gkf = GroupKFold(n_splits=folds, random_state=0, shuffle=True)
    k = 0
    for train_index, test_index in gkf.split(data, groups=data["PATNO"]):
        train_df = data.iloc[train_index]
        test_df = data.iloc[test_index]
        patno_train = train_df["PATNO"].unique()

        # follow up filtering in test set
        if filter_follow_up > 0:
            if verbose:
                print(
                    f"Filtering test set patients with follow-up cutoff {filter_follow_up}"
                )
            test_df = filter_follow_up_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                time_col=time_col,
                cutoff=filter_follow_up,
            )
            if verbose:
                print(
                    f"Number of patients in test set after filtering: {test_df['PATNO'].nunique()}"
                )

        # shorten follow-up in test set
        if shorten_follow_up < np.inf:
            if verbose:
                print(f"Shortening follow-up in test set to cutoff {shorten_follow_up}")
            test_df = shorten_follow_up_to_cutoff(
                data=test_df,
                timepoint_col=time_col,
                cutoff=shorten_follow_up,
            )
            if verbose:
                print(
                    f"Number of data points in test set after shortening: {test_df.shape[0]} and max timepoint is "
                    f"{test_df[time_col].max()}"
                )

        # filter missings in test set
        if filter_missings_cutoff < 1:
            if verbose:
                print(
                    f"Filtering test set patients with missingness cutoff {filter_missings_cutoff}"
                )
            test_df = filter_missings_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                scores=scores,
                cutoff=filter_missings_cutoff,
            )
            if verbose:
                print(
                    f"Number of patients in test set after filtering: {test_df['PATNO'].nunique()}"
                )

        # introduce missingness to test set
        if introduce_missingness > 0:
            if verbose:
                print(
                    f"Introducing missingness of {introduce_missingness} to test set patients. Missingness before is {test_df[scores].isna().mean().mean()}"
                )
            test_df = introduce_missingness_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                scores=scores,
                cutoff=introduce_missingness,
            )
            if verbose:
                print(
                    f"Missingness after introduction is {test_df[scores].isna().mean().mean()}"
                )

        # calculate likelihood ratio for each patient and score in the test set
        patno_test = test_df["PATNO"].unique()
        for patno in tqdm(
            patno_test,
            disable=not progress_bar,
            desc="Calculating likelihood ratios for fold k=" + str(k),
        ):
            if verbose:
                print("Processing patient:", patno)

            data_pat = test_df[test_df["PATNO"] == patno]

            likelihood_ratios[patno] = {}

            for score in scores:
                if verbose:
                    print("  Processing score:", score)
                # get first non-na score value
                data_pat_score = data_pat[[time_col, score]].dropna()

                # save NAN if we don't find a valid likelihood ratio
                likelihood_ratios[patno][score] = np.nan

                if verbose:
                    print(
                        "    Number of available score points:", data_pat_score.shape[0]
                    )
                    print("    Data points:", data_pat_score)

                if not data_pat_score.empty:
                    # take first score for which enough training data is available
                    for i in range(data_pat_score.shape[0]):
                        pat_time = data_pat_score.iloc[i][time_col]
                        pat_value = data_pat_score.iloc[i][score]
                        if verbose:
                            print(
                                "    Trying time:", pat_time, "with value:", pat_value
                            )

                        # round pat_time to same value as the grid
                        pat_time = np.round(pat_time)

                        # filter training dataset, current score, and current time
                        data_interpolated_train_score = data_interpolated[score]
                        data_interpolated_train_score = data_interpolated_train_score[
                            data_interpolated_train_score["PATNO"].isin(patno_train)
                        ]
                        data_interpolated_train_score = data_interpolated_train_score[
                            data_interpolated_train_score[time_col] == pat_time
                        ]

                        likelihood_ratio_log10 = np.log10(
                            calc_likelihood_ratio(
                                df=data_interpolated_train_score,
                                score_col=score,
                                new_score=pat_value,
                                subtype_col=subtype_col,
                                method=method,
                                enable_warnings=verbose,
                            )
                        )

                        if verbose:
                            print(
                                "    Calculated log likelihood ratio:",
                                likelihood_ratio_log10,
                            )

                        if first_score and not np.isnan(likelihood_ratio_log10):
                            likelihood_ratios[patno][score] = likelihood_ratio_log10
                            if verbose:
                                print("    - breaking after first value")
                            break

                        if not first_score:
                            if np.isnan(likelihood_ratio_log10):
                                if verbose:
                                    print(
                                        "    - log10 likelihood ratio is NaN, trying next value"
                                    )
                            else:
                                if np.isnan(likelihood_ratios[patno][score]):
                                    likelihood_ratios[patno][
                                        score
                                    ] = likelihood_ratio_log10
                                    if verbose:
                                        print("    - saved log10 likelihood ratio")
                                else:
                                    likelihood_ratios[patno][
                                        score
                                    ] += likelihood_ratio_log10
                                    if verbose:
                                        print(
                                            "    - added log10 likelihood ratio -> new value:",
                                            likelihood_ratios[patno][score],
                                        )
            if verbose:
                print("  Finished patient:", patno)
                print("  Final likelihood ratios:", likelihood_ratios[patno])
                print("-----------------------------------")
        k += 1

    likelihood_ratios = pd.DataFrame(likelihood_ratios).transpose()

    # calculate total log10 likelihood ratio
    likelihood_ratios["log10_lr_total"] = likelihood_ratios.replace(0, np.nan).sum(
        axis=1
    )

    # add subtype information
    subtypes = data.groupby("PATNO")["Subtype"].first().to_dict()
    likelihood_ratios["Subtype"] = likelihood_ratios.index.map(subtypes)

    return likelihood_ratios


def calc_likelihood_ratios_abs_train_test(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    first_score: bool,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate log 10 likelihood ratios for each patient using absolute scores and train/test datasets.

    Calculates log10 likelihood ratios for each patient based on absolute score values using a training dataset.
    Likelihood ratios are computed for each score at the first available timepoint (or all timepoints if specified),
    and the total log10 likelihood ratio is obtained by summing individual score log10 likelihood ratios across scores.

    Parameters
    ----------
    data_train : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Training dataset)
    data_test : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Test dataset)
    first_score : bool
        Whether to use only the first timepoint for each score for likelihood calculation or all available timepoints
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 1
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set. Needs to be a float between 0 and 1.
         Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from
        SCORE_LABELS if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with log10 likelihood ratios for each score and the total log10 likelihood ratio for the test
        dataset.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    additional_cols = [subtype_col]
    if scores is None:
        scores = list(SCORE_LABELS.keys())
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)

    def _interpolate_scores_worker(
        score: str,
        df: pd.DataFrame,
        time_col: str,
        patno_col: str,
        additional_cols: list[str],
        subtype_col: str,
    ) -> tuple[str, pd.DataFrame]:

        data_interpolated_score = interpolate_scores_to_grid(
            df=df,
            score_col=score,
            time_col=time_col,
            patno_col=patno_col,
            additional_cols=additional_cols,
        )

        # filter sparse timepoints
        data_interpolated_score = filter_sparse_timepoints(
            df=data_interpolated_score,
            score_col=score,
            time_col=time_col,
            subtype_col=subtype_col,
        )

        return score, data_interpolated_score

    # Calculate interpolated scores for all scores and patients in parallel
    # Each score is fitted in an isolated subprocess which is destroyed afterwards to avoid memory leaks

    # filter some warning messages that don't affect the results

    warnings.filterwarnings(
        "ignore",
        message=r"A worker stopped while some jobs were given to the executor.*",
        category=UserWarning,
        module=r"joblib\.externals\.loky\.process_executor",
    )

    # parallel execution
    results = Parallel(n_jobs=n_jobs, timeout=9999, backend="loky")(
        delayed(_interpolate_scores_worker)(
            score, data_train, time_col, patno_col, additional_cols, subtype_col
        )
        for score in tqdm(
            scores,
            disable=not progress_bar,
            desc=f"Submitting score interpolation jobs",
        )
    )

    # Convert results list to dictionary
    data_interpolated = dict(results)

    # prepare likelihood dataframe
    likelihood_ratios = {}

    patno_test = data_test[patno_col].unique()

    # follow up filtering in test set
    if filter_follow_up > 0:
        if verbose:
            print(
                f"Filtering test set patients with follow-up cutoff {filter_follow_up}"
            )
        data_test = filter_follow_up_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            time_col=time_col,
            cutoff=filter_follow_up,
        )
        if verbose:
            print(
                f"Number of patients in test set after filtering: {data_test['PATNO'].nunique()}"
            )

    # shorten follow-up in test set
    if shorten_follow_up < np.inf:
        if verbose:
            print(f"Shortening follow-up in test set to cutoff {shorten_follow_up}")
        data_test = shorten_follow_up_to_cutoff(
            data=data_test,
            timepoint_col=time_col,
            patno_col=patno_col,
            cutoff=shorten_follow_up,
        )
        if verbose:
            print(
                f"Number of data points in test set after shortening: {data_test.shape[0]} and max timepoint is "
                f"{data_test[time_col].max()}"
            )

    # filter missings in test set
    if filter_missings_cutoff < 1:
        if verbose:
            print(
                f"Filtering test set patients with missingness cutoff {filter_missings_cutoff}"
            )
        data_test = filter_missings_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            scores=scores,
            cutoff=filter_missings_cutoff,
        )
        if verbose:
            print(
                f"Number of patients in test set after filtering: {data_test['PATNO'].nunique()}"
            )

    # introduce missingness to test set
    if introduce_missingness > 0:
        if verbose:
            print(
                f"Introducing missingness of {introduce_missingness} to test set patients. "
                f"Missingness before is {data_test[scores].isna().mean().mean()}"
            )
        data_test = introduce_missingness_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            scores=scores,
            cutoff=introduce_missingness,
        )
        if verbose:
            print(
                f"Missingness after introduction is {data_test[scores].isna().mean().mean()}"
            )

    # calculate likelihood ratio for each patient and score in the test set
    for patno in tqdm(
        patno_test,
        disable=not progress_bar,
        desc="Calculating likelihood ratios for test set",
    ):
        if verbose:
            print("Processing patient:", patno)

        data_pat = data_test[data_test[patno_col] == patno]

        likelihood_ratios[patno] = {}

        for score in scores:
            if verbose:
                print("  Processing score:", score)
            # get first non-na score value
            data_pat_score = data_pat[[time_col, score]].dropna()

            # save NAN if we don't find a valid likelihood ratio
            likelihood_ratios[patno][score] = np.nan

            if verbose:
                print("    Number of available score points:", data_pat_score.shape[0])
                print("    Data points:", data_pat_score)

            if not data_pat_score.empty:
                # take first score for which enough training data is available
                for i in range(data_pat_score.shape[0]):
                    pat_time = data_pat_score.iloc[i][time_col]
                    pat_value = data_pat_score.iloc[i][score]
                    if verbose:
                        print("    Trying time:", pat_time, "with value:", pat_value)

                    # round pat_time to same value as the grid
                    pat_time = np.round(pat_time)

                    # filter training dataset, current score, and current time; in comparison to the cv method, we don't
                    # need to filter for training patient IDs as we have a separate training dataset here
                    data_interpolated_train_score = data_interpolated[score]
                    data_interpolated_train_score = data_interpolated_train_score[
                        data_interpolated_train_score[time_col] == pat_time
                    ]

                    likelihood_ratio_log10 = np.log10(
                        calc_likelihood_ratio(
                            df=data_interpolated_train_score,
                            score_col=score,
                            new_score=pat_value,
                            subtype_col=subtype_col,
                            method=method,
                            enable_warnings=verbose,
                        )
                    )

                    if verbose:
                        print(
                            "    Calculated log likelihood ratio:",
                            likelihood_ratio_log10,
                        )

                    if first_score and not np.isnan(likelihood_ratio_log10):
                        likelihood_ratios[patno][score] = likelihood_ratio_log10
                        if verbose:
                            print("    - breaking after first value")
                        break

                    if not first_score:
                        if np.isnan(likelihood_ratio_log10):
                            if verbose:
                                print(
                                    "    - log10 likelihood ratio is NaN, trying next value"
                                )
                        else:
                            if np.isnan(likelihood_ratios[patno][score]):
                                likelihood_ratios[patno][score] = likelihood_ratio_log10
                                if verbose:
                                    print("    - saved log10 likelihood ratio")
                            else:
                                likelihood_ratios[patno][
                                    score
                                ] += likelihood_ratio_log10
                                if verbose:
                                    print(
                                        "    - added log10 likelihood ratio -> new value:",
                                        likelihood_ratios[patno][score],
                                    )
        if verbose:
            print("  Finished patient:", patno)
            print("  Final likelihood ratios:", likelihood_ratios[patno])
            print("-----------------------------------")

    likelihood_ratios = pd.DataFrame(likelihood_ratios).transpose()

    # calculate total log10 likelihood ratio
    likelihood_ratios["log10_lr_total"] = likelihood_ratios.replace(0, np.nan).sum(
        axis=1
    )

    # add subtype information if available
    if subtype_col in data_test.columns:
        subtypes = data_test.groupby(patno_col)["Subtype"].first().to_dict()
        likelihood_ratios["Subtype"] = likelihood_ratios.index.map(subtypes)

    return likelihood_ratios


def calc_likelihood_ratios_slopes_cv(
    data: pd.DataFrame,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    folds: int = 10,
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate likelihood ratios for each patient using slopes and k-fold cross validation

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 1
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set in each fold. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set in each fold. Needs to be a float between 0 and 1.
         Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set in each fold. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set in each fold.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    folds : int, default 10
        Number of folds for k-fold cross validation
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from SCORE_LABELS
        if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with likelihood ratios for each score and the total log10 likelihood ratio.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    if scores is None:
        scores = list(SCORE_LABELS.keys())
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)

    # Worker function for fitting slope distribution in isolated process (to avoid memory leaks by statsmodels)
    def _fit_slope_distribution_worker(
        score: str,
        train_df: pd.DataFrame,
        time_col: str,
        patno_col: str,
        subtype_col: str,
    ) -> tuple[str, pd.DataFrame]:

        warnings.simplefilter("ignore", ConvergenceWarning)

        result = calc_score_slope_distribution(
            data=train_df,
            score_col=score,
            time_col=time_col,
            patno_col=patno_col,
            subtype_col=subtype_col,
        )
        return score, result

    # prepare likelihood dataframe
    likelihood_ratios = {}

    # k-fold split on patient level
    gkf = GroupKFold(n_splits=folds, random_state=0, shuffle=True)
    k = 0
    for train_index, test_index in gkf.split(data, groups=data[patno_col]):
        train_df = data.iloc[train_index]
        test_df = data.iloc[test_index]

        # calculate training set slopes
        # Each score is fitted in an isolated subprocess that releases memory on exit

        # filter some warning messages that don't affect the results
        warnings.filterwarnings(
            "ignore",
            message=r"A worker stopped while some jobs were given to the executor.*",
            category=UserWarning,
            module=r"joblib\.externals\.loky\.process_executor",
        )

        # execute in parallel
        results = Parallel(n_jobs=n_jobs, timeout=9999, backend="loky")(
            delayed(_fit_slope_distribution_worker)(
                score, train_df, time_col, patno_col, subtype_col
            )
            for score in tqdm(
                scores,
                disable=not progress_bar,
                desc=f"Submitting slope distribution fitting jobs for fold k={k}",
            )
        )

        # Convert results list to dictionary
        slopes_distribution = dict(results)

        # filter follow up in test set
        if filter_follow_up > 0:
            if verbose:
                print(
                    f"Filtering test set patients with follow-up cutoff {filter_follow_up}"
                )
            test_df = filter_follow_up_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                time_col=time_col,
                cutoff=filter_follow_up,
            )
            if verbose:
                print(
                    f"Number of patients in test set after filtering: {test_df['PATNO'].nunique()}"
                )

        # shorten follow-up in test set
        if shorten_follow_up < np.inf:
            if verbose:
                print(f"Shortening follow-up in test set to cutoff {shorten_follow_up}")
            test_df = shorten_follow_up_to_cutoff(
                data=test_df,
                timepoint_col=time_col,
                patno_col=patno_col,
                cutoff=shorten_follow_up,
            )
            if verbose:
                print(
                    f"Number of data points in test set after shortening: {test_df.shape[0]} and max timepoint is "
                    f"{test_df[time_col].max()}"
                )

        # filter missings in test set
        if filter_missings_cutoff < 1:
            if verbose:
                print(
                    f"Filtering test set patients with missingness cutoff {filter_missings_cutoff}"
                )
            test_df = filter_missings_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                scores=scores,
                cutoff=filter_missings_cutoff,
            )
            if verbose:
                print(
                    f"Number of patients in test set after filtering: {test_df['PATNO'].nunique()}"
                )

        # introduce missingness to test set
        if introduce_missingness > 0:
            if verbose:
                print(
                    f"Introducing missingness of {introduce_missingness} to test set patients. Missingness before "
                    f"is {test_df[scores].isna().mean().mean()}"
                )
            test_df = introduce_missingness_by_cutoff(
                data=test_df,
                patno_col=patno_col,
                scores=scores,
                cutoff=introduce_missingness,
            )
            if verbose:
                print(
                    f"Missingness after introduction is {test_df[scores].isna().mean().mean()}"
                )

        # calculate likelihood ratio for each patient and score slope in the test set
        patno_test = test_df[patno_col].unique()
        for patno in tqdm(
            patno_test,
            disable=not progress_bar,
            desc="Calculating likelihood ratios for fold k=" + str(k),
        ):
            if verbose:
                print("Processing patient:", patno)

            data_pat = test_df[test_df[patno_col] == patno]

            likelihood_ratios[patno] = {}

            for score in scores:
                if verbose:
                    print("  Processing score:", score)

                # get first non-na score value
                data_pat_score = data_pat[[time_col, score]].dropna()

                # save NAN if we don't find a valid likelihood ratio
                likelihood_ratios[patno][score] = np.nan

                # if fewer than 2 scores are available, we can't calculate a slope
                if data_pat_score.shape[0] >= 2:
                    # calculate slope using linear regression
                    patient_slope = stats.linregress(
                        data_pat_score[time_col], data_pat_score[score]
                    )[0]

                    likelihood_ratio_log10 = np.log10(
                        calc_likelihood_ratio(
                            df=slopes_distribution[score],
                            score_col=score,
                            new_score=patient_slope,
                            subtype_col=subtype_col,
                            method=method,
                            enable_warnings=verbose,
                        )
                    )
                    likelihood_ratios[patno][score] = likelihood_ratio_log10

                    if verbose:
                        print("    Calculated slope:", patient_slope)
                        print(
                            "    Calculated log likelihood ratio:",
                            likelihood_ratio_log10,
                        )
                else:
                    if verbose:
                        print("    Not enough data points to calculate slope.")
        k += 1

    likelihood_ratios = pd.DataFrame(likelihood_ratios).transpose()

    # calculate total log10 likelihood ratio
    likelihood_ratios["log10_lr_total"] = likelihood_ratios.replace(0, np.nan).sum(
        axis=1
    )

    # add subtype information
    subtypes = data.groupby(patno_col)["Subtype"].first().to_dict()
    likelihood_ratios["Subtype"] = likelihood_ratios.index.map(subtypes)

    return likelihood_ratios


def calc_likelihood_ratios_slopes_train_test(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate likelihood ratios for each patient using slopes and training/test datasets.

    Parameters
    ----------
    data_train : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Training dataset)
    data_test : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Test dataset)
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 1
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set in each fold. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set. Needs to be a float between 0 and 1.
         Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from SCORE_LABELS
        if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with likelihood ratios for each score and patient in the test set and the total log10 likelihood
        ratio.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    if scores is None:
        scores = list(SCORE_LABELS.keys())
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)

    # Worker function for fitting slope distribution in isolated process (to avoid memory leaks by statsmodels)
    def _fit_slope_distribution_worker(
            score: str,
            train_df: pd.DataFrame,
            time_col: str,
            patno_col: str,
            subtype_col: str,
    ) -> tuple[str, pd.DataFrame]:

        warnings.simplefilter("ignore", ConvergenceWarning)

        result = calc_score_slope_distribution(
            data=train_df,
            score_col=score,
            time_col=time_col,
            patno_col=patno_col,
            subtype_col=subtype_col,
        )
        return score, result

    # calculate training set slopes
    # Each score is fitted in an isolated subprocess that releases memory on exit

    # filter some warning messages that don't affect the results
    warnings.filterwarnings(
        "ignore",
        message=r"A worker stopped while some jobs were given to the executor.*",
        category=UserWarning,
        module=r"joblib\.externals\.loky\.process_executor",
    )

    # execute in parallel
    results = Parallel(n_jobs=n_jobs, timeout=9999, backend="loky")(
        delayed(_fit_slope_distribution_worker)(
            score, data_train, time_col, patno_col, subtype_col
        )
        for score in tqdm(
            scores,
            disable=not progress_bar,
            desc=f"Submitting slope distribution fitting jobs",
        )
    )

    # Convert results list to dictionary
    slopes_distribution = dict(results)

    # prepare likelihood dataframe
    likelihood_ratios = {}

    # filter follow up in test set
    if filter_follow_up > 0:
        if verbose:
            print(
                f"Filtering test set patients with follow-up cutoff {filter_follow_up}"
            )
        data_test = filter_follow_up_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            time_col=time_col,
            cutoff=filter_follow_up,
        )
        if verbose:
            print(
                f"Number of patients in test set after filtering: {data_test['PATNO'].nunique()}"
            )

    # shorten follow-up in test set
    if shorten_follow_up < np.inf:
        if verbose:
            print(f"Shortening follow-up in test set to cutoff {shorten_follow_up}")
        data_test = shorten_follow_up_to_cutoff(
            data=data_test,
            timepoint_col=time_col,
            patno_col=patno_col,
            cutoff=shorten_follow_up,
        )
        if verbose:
            print(
                f"Number of data points in test set after shortening: {data_test.shape[0]} and max timepoint is "
                f"{data_test[time_col].max()}"
            )

    # filter missings in test set
    if filter_missings_cutoff < 1:
        if verbose:
            print(
                f"Filtering test set patients with missingness cutoff {filter_missings_cutoff}"
            )
        data_test = filter_missings_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            scores=scores,
            cutoff=filter_missings_cutoff,
        )
        if verbose:
            print(
                f"Number of patients in test set after filtering: {data_test['PATNO'].nunique()}"
            )

    # introduce missingness to test set
    if introduce_missingness > 0:
        if verbose:
            print(
                f"Introducing missingness of {introduce_missingness} to test set patients. Missingness before "
                f"is {data_test[scores].isna().mean().mean()}"
            )
        data_test = introduce_missingness_by_cutoff(
            data=data_test,
            patno_col=patno_col,
            scores=scores,
            cutoff=introduce_missingness,
        )
        if verbose:
            print(
                f"Missingness after introduction is {data_test[scores].isna().mean().mean()}"
            )

    # calculate likelihood ratio for each patient and score slope in the test set
    patno_test = data_test[patno_col].unique()
    for patno in tqdm(
        patno_test,
        disable=not progress_bar,
        desc="Calculating likelihood ratios for test set",
    ):
        if verbose:
            print("Processing patient:", patno)

        data_pat = data_test[data_test[patno_col] == patno]

        likelihood_ratios[patno] = {}

        for score in scores:
            if verbose:
                print("  Processing score:", score)

            # get first non-na score value
            data_pat_score = data_pat[[time_col, score]].dropna()

            # save NAN if we don't find a valid likelihood ratio
            likelihood_ratios[patno][score] = np.nan

            # if fewer than 2 scores are available, we can't calculate a slope
            if data_pat_score.shape[0] >= 2:
                # calculate slope using linear regression
                patient_slope = stats.linregress(
                    data_pat_score[time_col], data_pat_score[score]
                )[0]

                likelihood_ratio_log10 = np.log10(
                    calc_likelihood_ratio(
                        df=slopes_distribution[score],
                        score_col=score,
                        new_score=patient_slope,
                        subtype_col=subtype_col,
                        method=method,
                        enable_warnings=verbose,
                    )
                )
                likelihood_ratios[patno][score] = likelihood_ratio_log10

                if verbose:
                    print("    Calculated slope:", patient_slope)
                    print(
                        "    Calculated log likelihood ratio:",
                        likelihood_ratio_log10,
                    )
            else:
                if verbose:
                    print("    Not enough data points to calculate slope.")
                    print("    Number of available score points:", data_pat_score.shape[0])
                    print("    Data points:", data_pat_score)

    likelihood_ratios = pd.DataFrame(likelihood_ratios).transpose()

    # calculate total log10 likelihood ratio
    likelihood_ratios["log10_lr_total"] = likelihood_ratios.replace(0, np.nan).sum(
        axis=1
    )

    # add subtype information if available
    if subtype_col in data_test.columns:
        subtypes = data_test.groupby(patno_col)["Subtype"].first().to_dict()
        likelihood_ratios["Subtype"] = likelihood_ratios.index.map(subtypes)

    return likelihood_ratios


def calc_likelihood_ratios_combined_cv(
    data: pd.DataFrame,
    models: list[str] | None = None,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    folds: int = 10,
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate likelihood ratios for each patient using slopes and k-fold cross validation

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes.
    models : list[str]
        List of models to include in the combined likelihood ratio. Options are all model keys given in MODEL_LABELS
        (absolute_first, absolute_all, slopes, slopes+absolute_first, slopes+absolute_all). Default is all models.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 0
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set in each fold. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set in each fold. Needs to be float between 0 and 1.
        Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set in each fold. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set in each fold.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    folds : int, default 10
        Number of folds for k-fold cross validation
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from SCORE_LABELS
        if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with likelihood ratios for each score and the total log10 likelihood ratio.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    if models is None:
        models = list(MODEL_LABELS.keys())
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    if (
        ("slope" in models)
        or ("slopes+absolute_first" in models)
        or ("slopes+absolute_all" in models)
    ):
        likelihood_ratios_slopes = calc_likelihood_ratios_slopes_cv(
            data=data,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            folds=folds,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_slopes["model"] = "slopes"

    if ("absolute_first" in models) or ("slopes+absolute_first" in models):
        likelihood_ratios_abs_bl = calc_likelihood_ratios_abs_cv(
            data=data,
            first_score=True,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            folds=folds,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_abs_bl["model"] = "absolute_first"

    if ("absolute_all" in models) or ("slopes+absolute_all" in models):
        likelihood_ratios_abs_all = calc_likelihood_ratios_abs_cv(
            data=data,
            first_score=False,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            folds=folds,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_abs_all["model"] = "absolute_all"

    # create a combined model where log10 likelihoods (for scores) are added (+0 if nan): slopes + absolute_BL
    if "slopes+absolute_first" in models:
        likelihood_ratios_combined_bl = likelihood_ratios_slopes.copy()
        for patno in likelihood_ratios_combined_bl.index:
            for score in scores + ["log10_lr_total"]:
                lr_slope = likelihood_ratios_slopes.loc[patno, score]
                lr_abs = likelihood_ratios_abs_bl.loc[patno, score]
                if np.isnan(lr_slope) and not np.isnan(lr_abs):
                    lr_slope = 0
                if np.isnan(lr_abs) and not np.isnan(lr_slope):
                    lr_abs = 0
                likelihood_ratios_combined_bl.loc[patno, score] = lr_slope + lr_abs
        likelihood_ratios_combined_bl["model"] = "slopes+absolute_first"

    # create a combined model where log10 likelihoods (for scores) are added (+0 if nan): slopes + absolute_all
    if "slopes+absolute_all" in models:
        likelihood_ratios_combined_all = likelihood_ratios_slopes.copy()
        for patno in likelihood_ratios_combined_all.index:
            for score in scores + ["log10_lr_total"]:
                lr_slope = likelihood_ratios_slopes.loc[patno, score]
                lr_abs = likelihood_ratios_abs_all.loc[patno, score]
                if np.isnan(lr_slope) and not np.isnan(lr_abs):
                    lr_slope = 0
                if np.isnan(lr_abs) and not np.isnan(lr_slope):
                    lr_abs = 0
                likelihood_ratios_combined_all.loc[patno, score] = lr_slope + lr_abs
        likelihood_ratios_combined_all["model"] = "slopes+absolute_all"

    # combine selected models
    likelihood_ratios = []
    if "absolute_first" in models:
        likelihood_ratios.append(likelihood_ratios_abs_bl)
    if "absolute_all" in models:
        likelihood_ratios.append(likelihood_ratios_abs_all)
    if "slopes" in models:
        likelihood_ratios.append(likelihood_ratios_slopes)
    if "slopes+absolute_first" in models:
        likelihood_ratios.append(likelihood_ratios_combined_bl)
    if "slopes+absolute_all" in models:
        likelihood_ratios.append(likelihood_ratios_combined_all)
    likelihood_ratios = pd.concat(likelihood_ratios, axis=0)

    # sort by 1) index, 2) "model"
    likelihood_ratios = likelihood_ratios.sort_values("model").sort_index()

    return likelihood_ratios


def calc_likelihood_ratios_combined_train_test(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    models: list[str] | None = None,
    patno_col: str = "PATNO",
    subtype_col: str = "Subtype",
    time_col: str = "Disease_duration",
    filter_missings_cutoff: float = 1,
    introduce_missingness: float = 0,
    filter_follow_up: float = 0,
    shorten_follow_up: float = np.inf,
    method: str = "z-score",
    scores: list[str] | None = None,
    progress_bar: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate likelihood ratios for each patient using slopes and training/test datasets.

    Parameters
    ----------
    data_train : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Training dataset)
    data_test : pandas.DataFrame
        DataFrame containing columns for patient IDs, time, score values, and subtypes. (Test dataset)
    models : list[str]
        List of models to include in the combined likelihood ratio. Options are all model keys given in MODEL_LABELS
        (absolute_first, absolute_all, slopes, slopes+absolute_first, slopes+absolute_all). Default is all models.
    patno_col : str, default "PATNO"
        The column name for patient IDs in the DataFrame.
    subtype_col : str, default "Subtype"
        The column name for subtypes in the DataFrame.
    time_col : str, default "Disease_duration"
        The column name for time in the DataFrame.
    filter_missings_cutoff: float, default 0
        Maximum fraction of missing scores allowed for patients to be included in the test set. The fraction is
        calculated per patient, as average over all times given for the patient in the DataFrame and over all scores
        given in 'scores'. If patients exceed this fraction, they are excluded from the test set. However,
        they are still included in the training set. Needs to be between 0 and 1. Default is 1 (no filtering)
    introduce_missingness : float, default 0
        Fraction of total missingness that should be introduced artificially into the data for testing purposes.
        Missingness is calculated per patient, as average over all times given for the patient in the DataFrame and
        over all scores given in 'scores'. If the missing fraction of a
        patient is already higher than the specified fraction, no additional missingness is introduced. If the
        missing fraction is lower, values are randomly set to NaN until the specified missing fraction is reached.
        Missingness introduction is only applied to the test set. Needs to be float between 0 and 1.
        Default is 0 (no missingness introduced).
    filter_follow_up: float, default 0
        Cutoff value applied to the maximum Timepoint per patient. Patients with a maximum Timepoint below the
        specified cutoff are excluded from the test set. However, they are still included in the training
        set. Default is 0 (no filtering).
    shorten_follow_up: float, default np.inf
        Cutoff value applied to the Timepoint. Timepoints above this value are removed from the test set.
        However, they are still included in the training set. Default is np.inf (no shortening).
    method : str, default "z-score"
        The method to use for likelihood calculation. Options are "empirical_cdf", and "z-score". Default is "z-score".
    scores : list[str] | None, optional
        List of scores for which likelihood ratios should be calculated. Default values are retrieved from SCORE_LABELS
        if None is provided.
    progress_bar : bool, default False
        Whether to show a progress bar during processing.
    n_jobs : int, optional
        Number of parallel jobs to use for fitting slope distributions. Defaults to number of CPU cores -1 if None.
    verbose : bool, default False
        Whether to print detailed processing information for each patient.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with likelihood ratios for each score and the total log10 likelihood ratio for patients in the
        test set.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """

    # arguments checking
    if models is None:
        models = list(MODEL_LABELS.keys())
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    if (
        ("slope" in models)
        or ("slopes+absolute_first" in models)
        or ("slopes+absolute_all" in models)
    ):
        likelihood_ratios_slopes = calc_likelihood_ratios_slopes_train_test(
            data_train=data_train,
            data_test=data_test,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_slopes["model"] = "slopes"

    if ("absolute_first" in models) or ("slopes+absolute_first" in models):
        likelihood_ratios_abs_bl = calc_likelihood_ratios_abs_train_test(
            data_train=data_train,
            data_test=data_test,
            first_score=True,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_abs_bl["model"] = "absolute_first"

    if ("absolute_all" in models) or ("slopes+absolute_all" in models):
        likelihood_ratios_abs_all = calc_likelihood_ratios_abs_train_test(
            data_train=data_train,
            data_test=data_test,
            first_score=False,
            patno_col=patno_col,
            subtype_col=subtype_col,
            time_col=time_col,
            filter_missings_cutoff=filter_missings_cutoff,
            introduce_missingness=introduce_missingness,
            filter_follow_up=filter_follow_up,
            shorten_follow_up=shorten_follow_up,
            method=method,
            scores=scores,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        likelihood_ratios_abs_all["model"] = "absolute_all"

    # create a combined model where log10 likelihoods (for scores) are added (+0 if nan): slopes + absolute_BL
    if "slopes+absolute_first" in models:
        likelihood_ratios_combined_bl = likelihood_ratios_slopes.copy()
        for patno in likelihood_ratios_combined_bl.index:
            for score in scores + ["log10_lr_total"]:
                lr_slope = likelihood_ratios_slopes.loc[patno, score]
                lr_abs = likelihood_ratios_abs_bl.loc[patno, score]
                if np.isnan(lr_slope) and not np.isnan(lr_abs):
                    lr_slope = 0
                if np.isnan(lr_abs) and not np.isnan(lr_slope):
                    lr_abs = 0
                likelihood_ratios_combined_bl.loc[patno, score] = lr_slope + lr_abs
        likelihood_ratios_combined_bl["model"] = "slopes+absolute_first"

    # create a combined model where log10 likelihoods (for scores) are added (+0 if nan): slopes + absolute_all
    if "slopes+absolute_all" in models:
        likelihood_ratios_combined_all = likelihood_ratios_slopes.copy()
        for patno in likelihood_ratios_combined_all.index:
            for score in scores + ["log10_lr_total"]:
                lr_slope = likelihood_ratios_slopes.loc[patno, score]
                lr_abs = likelihood_ratios_abs_all.loc[patno, score]
                if np.isnan(lr_slope) and not np.isnan(lr_abs):
                    lr_slope = 0
                if np.isnan(lr_abs) and not np.isnan(lr_slope):
                    lr_abs = 0
                likelihood_ratios_combined_all.loc[patno, score] = lr_slope + lr_abs
        likelihood_ratios_combined_all["model"] = "slopes+absolute_all"

    # combine selected models
    likelihood_ratios = []
    if "absolute_first" in models:
        likelihood_ratios.append(likelihood_ratios_abs_bl)
    if "absolute_all" in models:
        likelihood_ratios.append(likelihood_ratios_abs_all)
    if "slopes" in models:
        likelihood_ratios.append(likelihood_ratios_slopes)
    if "slopes+absolute_first" in models:
        likelihood_ratios.append(likelihood_ratios_combined_bl)
    if "slopes+absolute_all" in models:
        likelihood_ratios.append(likelihood_ratios_combined_all)
    likelihood_ratios = pd.concat(likelihood_ratios, axis=0)

    # sort by 1) index, 2) "model"
    likelihood_ratios = likelihood_ratios.sort_values("model").sort_index()

    return likelihood_ratios


def compute_grouped_roc_auc(
    df: pd.DataFrame,
    grouping_cols: str | list[str] | None = None,
    subtype_col: str = "Subtype",
    positive_subtype: int | str = SUBTYPE_FAST,
    score_col: str = "log10_lr_total",
    long_format: bool = True,
):
    """Compute ROC-AUC for predicting a binary subtype across groups using log10 likelihood ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing subtype labels and log10 likelihood ratios.
    grouping_cols : str or list of str or None, optional
        Column(s) to group by.
    subtype_col : str, default="Subtype"
        Column containing subtype labels.
    positive_subtype : int or str, default=SUBTYPE_FAST
        Value in subtype_col considered the positive class.
    score_col : str, default="log10_lr_total"
        Column containing prediction scores (e.g. log10_lr_total).
    long_format : bool, default=True
        If True, return long-format DataFrame with grouping columns + roc_auc.
        If False, return pivot-format DataFrame with grouping columns as index.

    Returns
    -------
    pd.DataFrame | float
        ROC-AUC results in long or pivot format if grouping_cols is provided, else a single float value.
    """

    if isinstance(grouping_cols, str):
        grouping_cols = [grouping_cols]

    def _calc_roc_auc(group):
        return roc_auc_score(
            y_true=(group[subtype_col] == positive_subtype).astype(int),
            y_score=group[score_col],
        )

    if grouping_cols is None or len(grouping_cols) == 0:
        df = df.dropna(subset=[subtype_col, score_col])
        return _calc_roc_auc(df)
    else:
        df = df.dropna(subset=[subtype_col, score_col] + grouping_cols)
        roc_df = (
            df.groupby(grouping_cols)[grouping_cols + [subtype_col, score_col]]
            .apply(_calc_roc_auc)
            .reset_index(name="roc_auc")
        )

    if long_format:
        return roc_df
    else:
        return roc_df.pivot(
            index=grouping_cols[0],
            columns=grouping_cols[1:] if len(grouping_cols) > 1 else None,
            values="roc_auc",
        )
