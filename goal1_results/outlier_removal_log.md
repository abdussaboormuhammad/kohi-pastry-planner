# Outlier Removal Log — Goal 1 Rebuild

Method: per-category MAD modified z-score on `daily_count`, threshold |M| > 3.5
(Iglewicz & Hoaglin 1993). Numeric predictors untouched.

| Category | Rows before | Removed | Removed % | Median | MAD | Removed values |
|---|---|---|---|---|---|---|
| Butter Croissant | 577 | 3 | 0.5% | 5.0 | 2.0 | 16, 25 |
| Chocolate Croissant | 550 | 2 | 0.4% | 3.0 | 1.0 | 9, 10 |
| Cookie/Brownie | 477 | 1 | 0.2% | 2.0 | 1.0 | 9 |
| Morning Bun | 249 | 1 | 0.4% | 2.0 | 1.0 | 11 |
| Muffin | 555 | 7 | 1.3% | 4.0 | 2.0 | 15, 16, 21, 22, 26 |
| Overnight Oats | 576 | 2 | 0.3% | 8.0 | 4.0 | 30 |
| Savory | 430 | 11 | 2.6% | 3.0 | 1.0 | 9, 10, 11, 12, 14 |
| Sweet Treat | 504 | 3 | 0.6% | 2.0 | 1.0 | 8, 9 |
| Yogurt Parfait | 565 | 8 | 1.4% | 4.0 | 2.0 | 15, 16, 17, 18 |

**Total: 38 of 4483 rows removed (0.8%). Original preserved as `data/pastry_kohi_raw.csv`.**
