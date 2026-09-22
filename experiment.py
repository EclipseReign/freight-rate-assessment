"""Chronological model selection, then one untouched September-October holdout."""
import json
from pathlib import Path
import pandas as pd
from modeling import FreightModel, metrics

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    'robust_ridge_core': dict(kind='ridge'),
    'robust_ridge_market': dict(kind='ridge', market=True),
    'catboost_core': dict(kind='catboost'),
    'catboost_market': dict(kind='catboost', market=True),
    'catboost_signal': dict(kind='catboost', market=True, signal=True),
    'robust_ridge_simple_core': dict(kind='ridge', simple_time=True),
    'robust_ridge_simple_market': dict(kind='ridge', market=True, simple_time=True),
}

def main():
    d = pd.read_csv(ROOT / 'train-test.csv')
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    candidates = CANDIDATES
    results = []
    for cutoff, end in [('2025-05-01', '2025-07-01'), ('2025-07-01', '2025-09-01')]:
        train = d[d.date < cutoff]
        test = d[(d.date >= cutoff) & (d.date < end)]
        for name, options in candidates.items():
            model = FreightModel(**options).fit(train)
            row = dict(model=name, start=cutoff, end=end, **metrics(test.posted_rate, model.predict(test)))
            results.append(row)
            print(row, flush=True)
        rpm = (train.posted_rate / train.distance).median()
        for name, pred in [('median_rpm', test.distance * rpm), ('quote_times_distance', test.distance * test.quote_signal)]:
            results.append(dict(model=name, start=cutoff, end=end, **metrics(test.posted_rate, pred)))
    finalize(d, candidates, results, output)


def finalize(d, candidates, results, output):
    pd.DataFrame(results).to_csv(output / 'model_selection.csv', index=False)
    scores = pd.DataFrame(results).groupby('model').mae.mean()
    primary = min(candidates, key=lambda key: scores[key])
    core = min([key for key, opts in candidates.items() if not opts.get('market')], key=lambda key: scores[key])
    selection = dict(primary=primary, core=core, configurations=candidates,
        selection_rule='Lowest mean dollar MAE across two disjoint two-month temporal folds; core restricted to chart-available features.')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    train, test = d[d.date < '2025-09-01'], d[d.date >= '2025-09-01']
    holdout = []
    for name in dict.fromkeys([primary, core]):
        model = FreightModel(**candidates[name]).fit(train)
        pred = model.predict(test)
        pd.DataFrame(dict(load_id=test.load_id, date=test.date, actual=test.posted_rate, predicted=pred)).to_csv(output / f'holdout_{name}.csv', index=False)
        for month in ['all', '2025-09', '2025-10']:
            mask = test.date.str.startswith(month) if month != 'all' else pd.Series(True, index=test.index)
            holdout.append(dict(model=name, month=month, **metrics(test.loc[mask, 'posted_rate'], pred[mask])))
    rpm = (train.posted_rate / train.distance).median()
    for name, pred in [('median_rpm', test.distance * rpm), ('quote_times_distance', test.distance * test.quote_signal)]:
        holdout.append(dict(model=name, month='all', **metrics(test.posted_rate, pred)))
    pd.DataFrame(holdout).to_csv(output / 'holdout_metrics.csv', index=False)
    print('SELECTION', selection, 'HOLDOUT', holdout, flush=True)

if __name__ == '__main__':
    main()
