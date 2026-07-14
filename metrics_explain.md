# Metrics Explanation — Goal 1: Kohi Pastry Demand Forecasting

## RMSE — Root Mean Squared Error
**Formula**: √mean((predicted − actual)²)

Primary ranking metric. Errors are squared before averaging, so large misses
(e.g., under-baking by 15 units on a busy event day) are penalised far more
than small ones. This aligns with the business cost of stock-outs.

**Guide**: An RMSE of 3.5 means predictions are off by roughly 3.5 units,
with larger errors weighted more heavily. Lower is better.

---

## MAE — Mean Absolute Error
**Formula**: mean(|predicted − actual|)

Every error is weighted equally — no squaring. An MAE of 2.0 means the model
is typically within 2 units of the actual count. Easier to communicate to the
pastry chef than RMSE. Used as first tiebreaker when RMSE values are equal.

---

## R² — Coefficient of Determination
**Formula**: 1 − SS_res / SS_tot

Proportion of variance in daily_count explained by the model.

| Range | Interpretation |
|-------|----------------|
| ≥ 0.70 | Strong — recommended for production use |
| 0.30–0.69 | Acceptable — useful for directional guidance |
| < 0.30 | Weak — little better than predicting the daily average |
| < 0 | Model's test error exceeds that of predicting the test-set mean |

**Note on negative values**: out-of-sample R² is *not* bounded at 0. The
metric compares the model's squared error against a baseline that predicts
the *test set's own mean* — a baseline the model never sees. On small
(~60–90 row), noisy test sets, a model can easily do worse than that
baseline, producing a negative R² even when the formula and pipeline are
correct.

---

## MAPE — Mean Absolute Percentage Error
**Formula**: mean(|predicted − actual| / actual) × 100

Volume-independent: a 3-unit miss on an item selling 10/day (30%) is correctly
flagged as worse than the same miss on an item selling 50/day (6%).
Enables fair cross-category comparison.

**Caveat**: Undefined when actual = 0. Days with zero sales are excluded;
result shown as N/A if no non-zero test days exist.

---

## Median AE — Median Absolute Error
**Formula**: median(|predicted − actual|)

Half of predictions fall within this distance of the actual count.
Robust to outlier event-day spikes that inflate MAE and RMSE.

---

## Ranking Priority
1. **RMSE ↑** — minimise large errors (primary)
2. **MAE ↑** — minimise typical error magnitude (tiebreaker)
3. **R² ↓** — prefer higher explanatory power (secondary tiebreaker)
