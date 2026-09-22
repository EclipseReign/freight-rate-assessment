# Loom walkthrough: approximately 2 minutes 40 seconds

Record this yourself while showing the actual files. This document is a script, not a video or a Loom link. The English narration below is approximately 340 words; rehearse at 125-140 words per minute. Pause recording if you need to navigate. Explain the code in your own words and only claim work you can discuss.

## 0:00-0:35 - Problem and exploration

Show the report's data-quality page and signal chart.

> This solution predicts a total freight rate for twelve thousand loads. The development data covers January through October 2025, while the final inputs cover November and December. I found missing and negative weights, extreme observed rates, eight cities absent from training, and new routes. An especially important finding is that the quote signal changes its relationship with the target across months. It cannot be treated as a consistently reliable price estimate.

## 0:35-1:10 - Model and cleaning

Show `modeling.py`: `prepare`, `fit`, and `predict`.

> The selected model is robust ridge regression on rate per mile. It uses shipment features, cities, equipment, coordinates, and smooth annual and weekly date features. The result is multiplied by distance to obtain a dollar price. Invalid weights become missing, and all medians and encoders are fitted on training rows only. Extreme responses are retained, but robust weights limit their training influence. Neither the identifier nor the target is an input feature. Unknown cities are supported.

## 1:10-1:55 - Validation and results

Show `experiment.py`, then the report's evaluation page.

> I compared seven configurations, including CatBoost and models with market and quote signals. Two chronological selection folds predict May-June and July-August from earlier data. The simple core model achieved the lowest average dollar MAE, about one hundred twenty-four dollars. I then froze the choice and evaluated September-October. Its MAE was one hundred sixty-nine dollars and forty-seven cents, about thirty-four percent better than the median-rate-per-mile baseline. WAPE was seven point zero eight percent. October was harder and predictions were biased low, so these results do not guarantee future accuracy. I did not tune on this final holdout.

## 1:55-2:40 - Outputs and December

Show `train_predict.py`, the prediction CSV header, and `scorer_results/candidate_december.png`.

> The final model is refitted on all labeled data. Predictions are matched to the template by load ID. For the fixed December scenario, city coordinates come from training data; the selected model needs neither the unavailable quote signal nor the market index. The original scorer validates all twelve thousand predictions and thirty-one scenario rows and generates this chart. It does not calculate hidden accuracy. The repository includes reproducible commands, pinned dependencies, validation results, and focused tests. The PDF documents both the findings and the limitations.

## Before sending

- Verify the assessor can open the GitHub repository and Loom recording.
- Attach `validation_predictions.csv` and `output/pdf/freight_rate_report.pdf`.
- Include the completed December CSV and chart in the repository.
- Never describe the local scorer as a hidden-set performance score.
