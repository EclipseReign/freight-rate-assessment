"""Build the English submission report and auditable dataset diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
INK = '#073E49'

def main():
    a = ROOT / 'artifacts'
    d = pd.read_csv(ROOT / 'train-test.csv')
    v = pd.read_csv(ROOT / 'validation.csv')
    selection = json.loads((a / 'selection.json').read_text())
    results = pd.read_csv(a / 'model_selection.csv')
    holdout = pd.read_csv(a / 'holdout_metrics.csv')
    final = pd.read_csv(ROOT / 'validation_predictions.csv')
    dec = pd.read_csv(ROOT / 'data/december_chart_inputs.csv')
    train_cities = set(d.pickup) | set(d.delivery)
    unseen = sorted((set(v.pickup) | set(v.delivery)) - train_cities)
    audit = dict(train_rows=len(d), validation_rows=len(v),
        train_dates=[d.date.min(), d.date.max()], validation_dates=[v.date.min(), v.date.max()],
        train_missing=d.isna().sum().to_dict(), validation_missing=v.isna().sum().to_dict(),
        train_negative_weight=int(d.weight.lt(0).sum()), validation_negative_weight=int(v.weight.lt(0).sum()),
        train_duplicate_ids=int(d.load_id.duplicated().sum()), validation_duplicate_ids=int(v.load_id.duplicated().sum()),
        duplicate_train_features_and_target=int(d.drop(columns='load_id').duplicated().sum()),
        unseen_cities=unseen, unseen_lane_rows=int((~(v.pickup+'|'+v.delivery).isin(d.pickup+'|'+d.delivery)).sum()),
        rates_above_5_per_mile=int((d.posted_rate/d.distance).gt(5).sum()))
    (a / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    d['rpm'] = d.posted_rate / d.distance
    monthly = []
    for month, rows in d.groupby(d.date.str[:7]):
        monthly.append(dict(month=month, n=len(rows), median_rpm=rows.rpm.median(),
            quote_spearman=rows.rpm.corr(rows.quote_signal, method='spearman'),
            quote_mae=np.mean(abs(rows.posted_rate-rows.distance*rows.quote_signal))))
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(a / 'monthly_signal_diagnostics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), dpi=180)
    axes[0].bar(monthly.month.str[5:], monthly.quote_spearman, color=INK)
    axes[0].axhline(0, color='gray', lw=.7)
    axes[0].set(title='Quote signal changes direction', xlabel='2025 month', ylabel='Spearman correlation with rate/mile', ylim=(-1.1, 1.1))
    axes[1].hist(d.rpm, bins=np.linspace(0, 15, 61), color=INK)
    axes[1].set(yscale='log', xlabel='Observed dollars per mile', ylabel='Loads (log scale)', title='Heavy-tailed observed responses')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(a / 'data_diagnostics.png')
    plt.close(fig)
    out = ROOT / 'output/pdf'
    out.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=27, leading=32, textColor=colors.HexColor(INK), spaceAfter=18))
    styles.add(ParagraphStyle(name='SectionCustom', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor(INK), spaceBefore=12, spaceAfter=10))
    styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10, leading=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='SmallCustom', fontName='Helvetica', fontSize=8, leading=11, spaceAfter=7))
    story = []
    def p(text, style='BodyCustom'):
        story.append(Paragraph(text, styles[style]))
    def title(kicker, text):
        p(kicker.upper(), 'SmallCustom')
        p(text, 'TitleCustom')
    def table(rows, widths=None):
        wrapped = [[Paragraph(str(c), styles['SmallCustom']) for c in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF0')),
            ('VALIGN',(0,0),(-1,-1),'TOP'), ('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('TOPPADDING',(0,0),(-1,-1),4), ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor(INK)),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D9E2E4'))]))
        story.append(t)
        story.append(Spacer(1, 12))
    winner = holdout[(holdout.model == selection['primary']) & (holdout.month == 'all')].iloc[0]
    baseline = holdout[(holdout.model == 'median_rpm') & (holdout.month == 'all')].iloc[0]
    title('Spotter / Machine Learning Engineer Assessment', 'Freight rate prediction')
    p('A reproducible forecast with chronological validation', 'SectionCustom')
    p(f"The final load model is <b>{selection['primary']}</b>. It was selected using two forward validation folds, then evaluated once on September-October 2025 before refitting on all 48,000 labeled loads. Model choices were frozen before that final evaluation.")
    table([['Final internal holdout', 'Result'],
        ['Training / evaluation', 'January-August / September-October 2025'],
        ['Evaluated loads', f'{int(winner.n):,}'],
        ['Mean absolute error (MAE)', f'${winner.mae:,.2f}'],
        ['Root mean square error (RMSE)', f'${winner.rmse:,.2f}'],
        ['Weighted absolute percentage error (WAPE)', f'{winner.wape:.2%}'],
        ['Median absolute error', f'${winner.median_ae:,.2f}'],
        ['Signed mean error (prediction - actual)', f'${winner.bias:,.2f}'],
        ['MAE improvement vs. training-median rate/mile', f'{1-winner.mae/baseline.mae:.1%}']], [310, 190])
    p('Delivered outputs', 'SectionCustom')
    p('The submission contains 12,000 positive load predictions, a completed 31-day December input file, the chart generated by the original scorer, executable training and validation code, fitted local models, diagnostic artifacts, and this report.')
    p('<b>What these scores mean:</b> they measure an internal chronological holdout, not the hidden November-December labels. The supplied score.py checks output contracts and creates a chart; it does not calculate forecast accuracy. External validation scores are unavailable.')
    p('All values in this report come from the supplied assessment files and the saved experiment outputs. No external data or hidden labels were used.', 'SmallCustom')
    story.append(PageBreak())

    title('01 / Exploration and cleaning', 'Data quality and distribution shift')
    table([['Check', 'Development', 'Final inputs'],
        ['Rows', '48,000', '12,000'], ['Date coverage', 'Jan 1-Oct 31, 2025', 'Nov 1-Dec 31, 2025'],
        ['Missing weight', str(d.weight.isna().sum()), str(v.weight.isna().sum())],
        ['Non-positive weight', str(d.weight.le(0).sum()), str(v.weight.le(0).sum())],
        ['Missing market index', str(d.market_index.isna().sum()), str(v.market_index.isna().sum())],
        ['Duplicate load IDs', '0', '0'],
        ['Cities absent from development', '-', str(len(unseen))],
        ['Rows on unseen directed lanes', '-', f"{audit['unseen_lane_rows']:,}"]], [250,125,125])
    p('Invalid weights are treated as missing, with an explicit missingness flag. Numerical medians are learned separately on each training fold; validation rows cannot alter them. Missing market indices receive the same training-only treatment in models that use them. No response outliers are deleted from training or evaluation.')
    p(f"There are {audit['rates_above_5_per_mile']} observed rates above $5/mile; the maximum posted rate is ${d.posted_rate.max():,.0f}. These observations are not assumed to be errors. Robust fitting reduces their influence while reported metrics retain every evaluation row.")
    story.append(Image(str(a / 'data_diagnostics.png'), width=500, height=155))
    p('The quote signal is strongly regime dependent: its association with rate/mile changes sign across months. A signal-using candidate is therefore tested explicitly rather than trusting the column name. Provided coordinates are treated as dataset features, without claiming verified real-world geography.', 'SmallCustom')
    p('Unseen cities: ' + ', '.join(unseen) + '. Unknown categories are supported, and supplied numerical coordinates remain available for these loads.', 'SmallCustom')
    story.append(PageBreak())

    title('02 / Evaluation design', 'Test the actual forecasting task')
    p('A random split would mix dates and signal regimes, making it a weak estimate of forecasting two future months. Each validation block instead covers two whole months, matching the final November-December horizon. No date is shared between training and evaluation within a fold.')
    table([['Stage', 'Training', 'Evaluation', 'Purpose'],
        ['Fold 1', 'Jan-Apr', 'May-Jun', 'Model selection'],
        ['Fold 2', 'Jan-Jun', 'Jul-Aug', 'Model selection'],
        ['Final holdout', 'Jan-Aug', 'Sep-Oct', 'One final estimate'],
        ['Final fit', 'Jan-Oct', 'Nov-Dec (unlabeled)', 'Submission']], [95,100,145,160])
    p('Candidates are ranked by the equal-weight mean of the two fold MAEs in dollars. MAE is chosen because the objective is an absolute load price and the response has extreme tails; the assessment does not specify an official metric. RMSE, WAPE, median absolute error and signed bias are also retained. No hyperparameter search or early stopping uses the final holdout.')
    rows = [['Candidate', 'May-Jun MAE', 'Jul-Aug MAE', 'Mean MAE']]
    for name, group in results.groupby('model', sort=False):
        vals = group.sort_values('start').mae.to_list()
        rows.append([name, f'${vals[0]:.2f}', f'${vals[1]:.2f}', f'${np.mean(vals):.2f}'])
    table(rows, [215,95,95,95])
    p('Untouched holdout results', 'SectionCustom')
    rows = [['Model / period', 'MAE', 'RMSE', 'WAPE']]
    for row in holdout.itertuples():
        rows.append([f'{row.model} / {row.month}', f'${row.mae:.2f}', f'${row.rmse:.2f}', f'{row.wape:.2%}'])
    table(rows, [245,85,85,85])
    p('WAPE = sum absolute error / sum actual rate. RMSE uses every residual, including extreme responses. Signed bias is mean(prediction - actual). Full-precision results and per-load holdout predictions are saved under artifacts/.', 'SmallCustom')
    p('The selected model underpredicts this holdout, and October is harder than September. This is evidence of remaining temporal shift, not a reason to tune on the final holdout. Its configuration is kept fixed for the final refit.', 'SmallCustom')
    story.append(PageBreak())

    title('03 / Model and reproducibility', 'A small, auditable pipeline')
    p('Features and response', 'SectionCustom')
    p('Models learn posted_rate / distance and convert predictions back to dollars using the supplied distance. Inputs include pickup, delivery, equipment, distance, log-distance, inverse distance, weight and its missingness flag, endpoint coordinates, coordinate differences, and annual and weekly Fourier terms. The selected simple-time model has one annual harmonic, two weekly harmonics and no linear trend; richer candidates add a second annual harmonic and trend. load_id and posted_rate are never input features. Market and quote signals are included only in explicitly named candidates.')
    p('Why compare these models?', 'SectionCustom')
    p('Robust ridge provides a smooth, regularized baseline with one-hot city/equipment effects and standardized numerical features. Fifteen iterations of residual-based Huber weighting limit extreme response influence; the residual scale is fitted only to training responses. Ridge alpha is fixed at 20. CatBoost offers nonlinear interactions and native categorical handling, using 700 trees, depth 6, learning rate 0.055, MAE loss and seed 42. Both fit rate/mile, while selection measures dollars.')
    p(f"The selected production model is <b>{selection['primary']}</b>; the reduced-input December model is <b>{selection['core']}</b>. The latter is selected from candidates that require neither market_index nor quote_signal. Selection is based entirely on forward folds. If the two names match, the same fitted estimator is used for both outputs.")
    p('Reproducibility and checks', 'SectionCustom')
    p('experiment.py reproduces model selection and holdout evaluation. train_predict.py refits the frozen choices, saves estimators, aligns predictions by load_id with the supplied template, and invokes the unmodified scorer. build_report.py rebuilds this report. Exact package versions and source SHA-256 hashes are recorded. The original assessment inputs remain unchanged.')
    p('Focused tests verify the official CSV contracts, prediction agreement after model serialization, ID/target exclusion, missing and invalid weights, unknown categories, batch-independent predictions, and preservation of all fixed December inputs. Final output is rounded to cents and must remain finite and positive.')
    p('Limits of the evidence', 'SectionCustom')
    p('Only ten labeled months from one year are available. Annual components extrapolate into November-December without a previous winter for validation. The December chart is a conditional point prediction, not a measured market series or calibrated interval. Missing weight can conceal useful information; new cities lack learned city-specific effects. A robust MAE-oriented estimate can underpredict rare expensive loads and is not optimized for an unspecified hidden RMSE objective.')
    story.append(PageBreak())

    title('04 / Required fixed-input scenario', 'December 2025 prediction chart')
    p('Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb')
    story.append(Image(str(ROOT / 'scorer_results/candidate_december.png'), width=500, height=500 * 4.8 / 10.8))
    story.append(Spacer(1,12))
    p('The image above is produced by the original score.py from data/december_chart_inputs.csv. All 31 original dates and fixed shipment fields are preserved; only predicted_rate is populated. No hand-drawn or post-processed rate curve is substituted.')
    p(f"The predicted range is <b>${dec.predicted_rate.min():,.2f}-${dec.predicted_rate.max():,.2f}</b>, with a monthly mean of <b>${dec.predicted_rate.mean():,.2f}</b>. The model uses the same shipment inputs on each day; variation comes from date features.")
    p('Handling the reduced schema', 'SectionCustom')
    p('The chart inputs omit coordinates, market_index and quote_signal. Endpoint coordinates are recovered from the median coordinate mapping in labeled training data only. The selected core model does not require the two unavailable signals, so no future market values or quotes are fabricated. Chart construction uses the final training fit, just like load prediction.')
    p('Submission status', 'SectionCustom')
    p(f"The original scorer accepts all {len(final):,} final predictions and all {len(dec)} December rows. The GitHub repository must be made accessible to the assessor, and the candidate must record and provide a genuine 2-3 minute Loom link. A timed recording script is supplied separately; no hosted repository or video URL is claimed by this report.")
    p('Source: supplied freight-rate-ml-assessment.pdf, readme.md, four CSV inputs and score.py. Numerical evidence: artifacts/model_selection.csv, holdout_metrics.csv, data_audit.json and monthly_signal_diagnostics.csv.', 'SmallCustom')

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#D9E2E4'))
        canvas.line(47, 42, A4[0]-47,42)
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(47,29,'SPOTTER  /  FREIGHT RATE PREDICTION')
        canvas.drawRightString(A4[0]-47,29,str(doc.page))
    doc = SimpleDocTemplate(str(out / 'freight_rate_report.pdf'), pagesize=A4, rightMargin=47,leftMargin=47,
        topMargin=42,bottomMargin=55, title='Freight Rate Prediction - Validation and Model Report',author='Assessment candidate')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out / 'freight_rate_report.pdf')

if __name__ == '__main__':
    main()
