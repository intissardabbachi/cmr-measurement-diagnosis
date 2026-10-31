# =============================================================================
# STEP 10 — Few-shot recalibration of the decision rule at a new site (CPU, ~3-5 min)
#
# Step 9c showed that 87% of external errors are RULE errors (wrong even with reference
# measurements) -> the ACDC decision thresholds do not match the new population.
# Because the decision is a linear rule on 21 named measurements, it can be re-anchored
# with a FEW labelled patients of the new site (diagnosis only, no manual contours):
#   A0  locked          : ACDC model, no adaptation
#   A1  intercept-only  : keep all 21x5 weights, re-fit the 3 class offsets (L2, lambda = 1)
#                         -> "the thresholds moved", weights stay readable
#   A2  joint refit     : re-fit the head on ACDC (reference measurements) + the k-shot
#                         patients (predicted measurements, sample weight 5), same C
# Protocol (fixed before running): per vendor, k labelled patients per class are drawn
# at random from that vendor (k = 1, 3, 5, 10), the rest of the vendor is the test set;
# 100 random draws per k. Measurements: step 9 cache + the ED/ES rule of step 9c.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.special import logsumexp, log_softmax
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
OUT = f'{DRIVE}/step9_mnms_external'
SEED, KS, DRAWS, LAM, W_NEW = 42, (1, 3, 5, 10), 100, 1.0, 5.0
PRIMARY = ['DCM', 'HCM', 'NOR']
rng = np.random.default_rng(SEED)

folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y_acdc = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(A_gt, y_acdc)
head, C_best = grid.best_estimator_, grid.best_params_['logisticregression__C']
pidx = np.searchsorted(classes, PRIMARY)
F = {c: j for j, c in enumerate(FEATS)}

# ---------------- M&Ms measurements with the ED/ES rule of step 9c
R = pd.DataFrame(list(np.load(f'{OUT}/mnms_concepts.npz', allow_pickle=True)['res']))
Cp = np.stack(list(R['pred']))
def swap_phases(X):
    X = X.copy()
    for a in ('RV', 'MYO', 'LV'):
        X[..., [F[f'{a}_ED'], F[f'{a}_ES']]] = X[..., [F[f'{a}_ES'], F[f'{a}_ED']]]
    for s in ('max', 'mean', 'std', 'min'):
        X[..., [F[f'WT_ED_{s}'], F[f'WT_ES_{s}']]] = X[..., [F[f'WT_ES_{s}'], F[f'WT_ED_{s}']]]
    e = 1e-6; g = lambda c: X[..., F[c]]
    X[..., F['LV_EF']] = 100 * (g('LV_ED') - g('LV_ES')) / np.maximum(g('LV_ED'), e)
    X[..., F['RV_EF']] = 100 * (g('RV_ED') - g('RV_ES')) / np.maximum(g('RV_ED'), e)
    X[..., F['LV_mass']] = 1.05 * g('MYO_ED')
    X[..., F['RV_LV_ED']] = g('RV_ED') / np.maximum(g('LV_ED'), e); X[..., F['RV_LV_ES']] = g('RV_ES') / np.maximum(g('LV_ES'), e)
    X[..., F['MYO_LV_ED']] = g('MYO_ED') / np.maximum(g('LV_ED'), e); X[..., F['MYO_LV_ES']] = g('MYO_ES') / np.maximum(g('LV_ES'), e)
    return X
sw = Cp[:, F['LV_ED']] < Cp[:, F['LV_ES']]
Cp[sw] = swap_phases(Cp[sw])
yv = np.searchsorted(PRIMARY, R.label.values); vend = R.vendor.values
L0 = head.decision_function(Cp)[:, pidx]                          # locked logits (3 primary classes)

def fit_offsets(L, yy):
    """A1: class offsets b (3,) on top of fixed logits, L2-penalised towards 0."""
    Y = np.eye(3)[yy]
    def obj(b):
        Z = L + b; P = np.exp(log_softmax(Z, 1))
        return -(Y * log_softmax(Z, 1)).sum() + LAM * (b ** 2).sum(), (P - Y).sum(0) + 2 * LAM * b
    return minimize(obj, np.zeros(3), jac=True, method='L-BFGS-B').x

def refit(idx):
    """A2: head re-fitted on ACDC (reference) + k-shot patients (predicted), weight W_NEW."""
    X = np.vstack([A_gt, Cp[idx]]); yy = np.concatenate([y_acdc, pidx[yv[idx]]])
    w = np.concatenate([np.ones(len(y_acdc)), np.full(len(idx), W_NEW)])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced', C=C_best))
    m.fit(X, yy, logisticregression__sample_weight=w)
    return m

rows, offs = [], []
for v in sorted(set(vend)):
    iv = np.where(vend == v)[0]
    present = sorted(set(yv[iv]))
    rows.append(dict(vendor=v, k=0, method='A0 locked', bal_acc=balanced_accuracy_score(yv[iv], L0[iv].argmax(1)), sd=0.0, n_test=len(iv)))
    for k in KS:
        if any((yv[iv] == c).sum() <= k for c in present):
            continue                                              # need >= 1 test patient per class
        acc = {'A0 locked': [], 'A1 intercept-only': [], 'A2 joint refit': []}
        for _ in range(DRAWS):
            ad = np.concatenate([rng.choice(iv[yv[iv] == c], k, replace=False) for c in present])
            te = np.setdiff1d(iv, ad)
            acc['A0 locked'].append(balanced_accuracy_score(yv[te], L0[te].argmax(1)))
            b = fit_offsets(L0[ad], yv[ad]); offs.append(dict(vendor=v, k=k, **dict(zip(PRIMARY, b - b.mean()))))
            acc['A1 intercept-only'].append(balanced_accuracy_score(yv[te], (L0[te] + b).argmax(1)))
            m2 = refit(ad)
            acc['A2 joint refit'].append(balanced_accuracy_score(yv[te], m2.decision_function(Cp[te])[:, pidx].argmax(1)))
        for meth, a in acc.items():
            rows.append(dict(vendor=v, k=k, method=meth, bal_acc=np.mean(a), sd=np.std(a), n_test=len(iv) - k * len(present)))
        print(f'{v}: k = {k} done')

T = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('\nBalanced accuracy on held-out patients of each vendor (mean +- SD over 100 draws):')
print(T.pivot_table(index=['vendor', 'k'], columns='method', values='bal_acc').round(3).to_string())

# vendor-weighted summary (only k available for all vendors that have it)
print('\nPooled over vendors (weighted by test size):')
for k in KS:
    t = T[T.k == k]
    if t.empty: continue
    print(f'  k = {k:2d}: ' + ' | '.join(f'{m}: {np.average(g.bal_acc, weights=g.n_test):.3f}' for m, g in t.groupby('method'))
          + f'  (vendors: {sorted(t.vendor.unique())})')

O = pd.DataFrame(offs)
print('\nHow the rule moved (A1 class offsets, centred, median over draws, k = 5): positive = class made more likely')
print(O[O.k == 5].groupby('vendor')[PRIMARY].median().round(2).to_string())

T.to_csv(f'{OUT}/step10_fewshot.csv', index=False); O.to_csv(f'{OUT}/step10_offsets.csv', index=False)
print('\nSaved to', OUT)
