"""About tab: background on the app, methodology, disclaimer."""
import streamlit as st


def render(*_):
    st.markdown("## About the Parkinson Subtype Predictor")

    st.markdown(
        """
        A web app that predicts Parkinson's disease progression subtype --
        fast or slow -- from the trajectories of clinical scores. It is a
        research and demonstration tool. The predictions are **not clinically
        validated** and do not replace medical judgment.
        """
    )

    st.markdown("### How the models work")
    st.markdown(
        """
        Three classifiers were trained on the PPMI cohort (Parkinson's
        Progression Markers Initiative, n=409 patients).

        - **Random Forest** -- ensemble of 500 decision trees
        - **XGBoost** -- gradient-boosted trees
        - **Logistic Regression** with L1 regularization

        For each clinical score we extract two features per patient: the
        **slope** (linear regression across all visits) and the
        **intercept** (extrapolated value at diagnosis). For patients with
        only one visit, a separate single-visit model is used that works on
        absolute score values -- less accurate, but better than nothing.

        All models are calibrated via isotonic cross-validation calibration,
        so the output probabilities are interpretable: 70% means roughly
        7 out of 10 comparable cases turn out to be of that class.
        """
    )

    st.markdown("### Accuracy on PPMI")
    g1, g2, g3 = st.columns(3)
    g1.metric("Random Forest AUC", "0.95")
    g2.metric("XGBoost AUC", "0.95")
    g3.metric("Logistic Regression AUC", "0.88")
    st.caption("On 10-fold cross-validation grouped by patient.")

    st.markdown("### Score sets")
    st.markdown(
        """
        - **Standard (17 scores)** -- clinical routine scores measured in
          most PD clinics. These 17 overlap with the LuxPARK cohort
          (Luxembourg) and form the basis of our external validation.
        - **Extended (25 scores)** -- adds the PPMI research battery
          (HVLT, SDM, LNS, VFT semantic, SEADL, ESS, GDS). Slightly higher
          accuracy, but rarely available in clinical routine.
        """
    )

    st.markdown("### Handling of missing data")
    st.markdown(
        """
        Missing score values are imputed with the median of the training
        cohort, so predictions are still possible even with patchy data. But
        reliability drops as missingness increases. To stay transparent, the
        app shows the **expected AUC** for each classifier at the current
        missingness and follow-up level, based on a simulation on the PPMI
        dataset.
        """
    )

    st.markdown("### Disclaimer")
    st.info(
        "This app is a research and demonstration tool. The predictions are "
        "not clinically validated and must not replace medical decision-making.",
        icon=":material/info:",
    )

    st.markdown("### Code and data")
    st.markdown(
        """
        - Training data: PPMI ([ppmi-info.org](https://www.ppmi-info.org))
        - Code: [github.com/cl-poehl/parkinson-subtype-predictor](https://github.com/cl-poehl/parkinson-subtype-predictor)
        - External validation in preparation on the LuxPARK cohort
        """
    )
