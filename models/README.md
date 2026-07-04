# Models are not distributed here

The trained model files are intentionally **excluded** from this public
repository. Beyond being large, they are participant-level derivatives of
PPMI data: the kNN imputer inside each pipeline retains the training feature
matrix, so the model files embed PPMI-derived participant data and fall under
the PPMI Data Use Agreement.

To obtain working models:

- **Use the live demo** (link in the top-level README), or
- **Regenerate them** from PPMI data with the training scripts:
  `python scripts/train_models.py`, `train_fallback_models.py`,
  `train_vennabers.py` (see the top-level README).
