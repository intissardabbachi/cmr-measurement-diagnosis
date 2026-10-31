# =============================================================================
# STEP 8b — Is the step-8 improvement real? Paired bootstrap over patients (B = 2000)
#   pp_cal (per-patient uncertainty) vs plain (step 2b), and pp_raw vs plain.
#   + exact tests for the "flip flag" (a measurement flips the decision in >= 20% of samples).
# Needs: step8_patient_uncertainty outputs + folds.csv. CPU, about a minute.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.stats import fisher_exact, beta
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S8 = f'{DRIVE}/step8_patient_uncertainty'
SEED, B, FLAG = 42, 2000, 0.20

folds = pd.read_csv(f'{DRIVE}/folds.csv')
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
n, K = len(y), len(classes)
P = {m: np.load(f'{S8}/oof_probs_{m}.npy') for m in ('plain', 'pp_raw', 'pp_cal')}
ex = pd.read_csv(f'{S8}/doubt_per_patient.csv').set_index('patient').loc[folds.patient]

def ece(Pm, yy, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == yy).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

NAMES = ['bal_acc', 'AUC', 'NLL', 'Brier', 'ECE', 'err_detect_AUROC']
def metrics(Pm, i):
    yy, Q = y[i], Pm[i]
    wrong = Q.argmax(1) != yy
    return np.array([balanced_accuracy_score(yy, Q.argmax(1)),
                     roc_auc_score(yy, Q, multi_class='ovr', labels=range(K)),
                     -np.mean(np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-12, 1))),
                     np.mean(((Q - np.eye(K)[yy]) ** 2).sum(1)),
                     ece(Q, yy),
                     roc_auc_score(wrong, 1 - Q.max(1)) if 0 < wrong.sum() < len(yy) else np.nan])

rng = np.random.default_rng(SEED)
boots = [i for i in (rng.integers(0, n, n) for _ in range(B)) if len(np.unique(y[i])) == K]
pt = {m: metrics(P[m], np.arange(n)) for m in P}
bs = {m: np.array([metrics(P[m], i) for i in boots]) for m in P}

pd.set_option('display.width', 200)
print('Point estimates [95% CI]:')
rows = []
for m in P:
    lo, hi = np.nanpercentile(bs[m], [2.5, 97.5], axis=0)
    rows.append({'model': m, **{k: f'{pt[m][j]:.3f} [{lo[j]:.3f}, {hi[j]:.3f}]' for j, k in enumerate(NAMES)}})
print(pd.DataFrame(rows).to_string(index=False))

print('\nPaired differences (model minus plain), 95% CI; * = CI excludes 0')
better = {'bal_acc': +1, 'AUC': +1, 'NLL': -1, 'Brier': -1, 'ECE': -1, 'err_detect_AUROC': +1}
drows = []
for m in ('pp_raw', 'pp_cal'):
    d = bs[m] - bs['plain']
    for j, k in enumerate(NAMES):
        lo, hi = np.nanpercentile(d[:, j], [2.5, 97.5])
        sig = '*' if lo > 0 or hi < 0 else ''
        direction = 'better' if np.sign(pt[m][j] - pt['plain'][j]) == better[k] else 'worse'
        drows.append(dict(model=m, metric=k, diff=pt[m][j] - pt['plain'][j], lo=lo, hi=hi, sig=sig, direction=direction,
                          p_better=np.nanmean(np.sign(d[:, j]) == better[k])))
D = pd.DataFrame(drows)
print(D.round(3).to_string(index=False))

# flip flag: exact statistics (errors of the plain model)
wrong = (ex.label != ex.pred).values
flag = (ex.doubt1_flip >= FLAG).values
a, b_ = (flag & wrong).sum(), (~flag & wrong).sum()
c, d_ = (flag & ~wrong).sum(), (~flag & ~wrong).sum()
def cp(k, m):                                                  # Clopper-Pearson 95%
    return (beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, m - k) if k < m else 1.0)
odds, pv = fisher_exact([[a, b_], [c, d_]], alternative='greater')
print(f'\nFlip flag (>= {FLAG:.0%} of samples flip the decision):')
print(f'  sensitivity (wrong flagged)  = {a}/{a + b_} = {a / (a + b_):.2f}  95% CI {np.round(cp(a, a + b_), 2)}')
print(f'  false-flag rate (correct)    = {c}/{c + d_} = {c / (c + d_):.2f}  95% CI {np.round(cp(c, c + d_), 2)}')
print(f'  precision (flagged are wrong) = {a}/{a + c} = {a / max(a + c, 1):.2f}')
print(f'  Fisher exact (one-sided) p = {pv:.2g}, odds ratio = {odds:.1f}')
print(f'  AUROC of max flip fraction for detecting errors = {roc_auc_score(wrong, ex.doubt1_flip.values):.3f}')

D.to_csv(f'{S8}/step8b_paired_differences.csv', index=False)
print('\nSaved to', S8)
