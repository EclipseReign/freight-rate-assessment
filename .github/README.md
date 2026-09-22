# Freight rate prediction

Reproducible solution to the Spotter Machine Learning Engineer assessment.

- [Solution and run instructions](../SOLUTION.md)
- [PDF report](../output/pdf/freight_rate_report.pdf)
- [12,000 load predictions](../validation_predictions.csv)
- [Completed December scenario](../data/december_chart_inputs.csv)
- [December chart](../scorer_results/candidate_december.png)
- [Validation metrics](../artifacts/holdout_metrics.csv)

The selected robust regression model achieved $169.47 MAE and 7.08% WAPE on the chronological September-October holdout. These are internal evaluation results; hidden-set accuracy is unavailable. The original scorer validates all 12,000 load predictions and all 31 December rows.

The original assessment datasets are not redistributed. See the solution instructions for the required local inputs and reproduction commands.
