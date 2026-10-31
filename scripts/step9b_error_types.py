# =============================================================================
# STEP 9b — Why does the model fail on M&Ms? Error decomposition (CPU, ~1-2 min)
#   Uses the cached M&Ms measurements of step 9 (nothing re-segmented, nothing re-fitted).
#   Numerical fix: the 3-class restriction is done in log space (step 9 divided
#   underflowed probabilities -> NaN for a few patients). Everything else identical.
#
#   For each patient, two decisions from the same locked ACDC model:
#     ref  = decision from reference (gt) measurements
#     pred = decision from predicted measurements (the real pipeline)
#   Error types (pred wrong):
#     SEG  : ref correct, pred wrong  -> error caused by segmentation/measurement
#     RULE : ref wrong,   pred wrong  -> error of the ACDC decision rule (population shift)
#   The flip flag is designed to see SEG errors, not RULE errors.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import fisher_exact, beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S8, OUT = f'{DRIVE}/step8_patient_uncertainty', f'{DRIVE}/step9_mnms_external'
SEED, S, FLAG, B = 42, 20, 0.20, 2000
PRIMARY = ['DCM', 'HCM', 'NOR']
rng = np.random.default_rng(SEED)

# locked ACDC model (identical to step 9)
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y_acdc = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
A_cs = np.load(f'{S8}/concept_samples.npz')['CS']
head = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(A_gt, y_acdc).best_estimator_
KAPPA = float(np.median(np.abs(A_pr - A_gt) / (A_cs.std(1) + 1e-9)) / 0.6745)
pidx = np.searchsorted(classes, PRIMARY)

def p3(X):                                                     # restriction in log space (no underflow)
    L = head.decision_function(X)[:, pidx]
    return np.exp(L - logsumexp(L, 1, keepdims=True))

R = pd.DataFrame(list(np.load(f'{OUT}/mnms_concepts.npz', allow_pickle=True)['res']))
n = len(R)
Cg, Cp, CS = np.stack(list(R['gt'])), np.stack(list(R['pred'])), np.stack(list(R['samples']))
yv = np.searchsorted(PRIMARY, R.label.values); vend = R.vendor.values
dv = CS - CS.mean(1, keepdims=True)
Pp, Pg = p3(Cp), p3(Cg)
pred, ref = Pp.argmax(1), Pg.argmax(1)
wrong, ref_ok = pred != yv, ref == yv
print(f'n = {n} | kappa = {KAPPA:.2f}')
print(f'bal acc (predicted measurements) = {balanced_accuracy_score(yv, pred):.3f} | '
      f'bal acc (reference measurements) = {balanced_accuracy_score(yv, ref):.3f}')

flips = np.zeros((n, len(FEATS)))
for t in range(n):
    for j in range(len(FEATS)):
        X = np.repeat(Cp[[t]], S, 0); X[:, j] += KAPPA * dv[t, :, j]
        flips[t, j] = np.mean(p3(X).argmax(1) != pred[t])
fmax = flips.max(1); flag = fmax >= FLAG; conf = 1 - Pp.max(1)
doubt = np.array(FEATS)[flips.argmax(1)]

typ = np.where(~wrong, 'CORRECT', np.where(ref_ok, 'SEG', 'RULE'))
rescued = (~ref_ok) & (~wrong)
print('\nError types (pred wrong):')
print(pd.crosstab(pd.Series(vend, name='vendor'), pd.Series(typ, name='type'), margins=True).to_string())
print(f'(also: {rescued.sum()} patients wrong with reference measurements but right with predicted ones)')

def cp(k, m):
    return (beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, m - k) if k < m else 1.0)
ok = typ == 'CORRECT'
print(f'\nFalse-flag rate on correct patients: {flag[ok].sum()}/{ok.sum()} = {flag[ok].mean():.2f} CI {np.round(cp(flag[ok].sum(), ok.sum()), 2)}')
rows = []
for tname in ('SEG', 'RULE'):
    m = typ == tname
    if m.sum() == 0: continue
    sel = m | ok; w = m[sel]
    odds, pv = fisher_exact([[flag[m].sum(), (~flag[m]).sum()], [flag[ok].sum(), (~flag[ok]).sum()]], alternative='greater')
    au_f, au_c = roc_auc_score(w, fmax[sel]), roc_auc_score(w, conf[sel])
    bs = []
    for _ in range(B):
        i = rng.choice(np.where(sel)[0], sel.sum())
        ww = m[i]
        if 0 < ww.sum() < len(i):
            bs.append(roc_auc_score(ww, fmax[i]) - roc_auc_score(ww, conf[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(type=tname, n=m.sum(), flag_sens=flag[m].mean(), sens_CI=np.round(cp(flag[m].sum(), m.sum()), 2),
                     Fisher_p=pv, OR=odds, AUROC_flip=au_f, AUROC_conf=au_c, diff=au_f - au_c, diff_lo=lo, diff_hi=hi))
T = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('\nDoes the flag catch each error type? (vs correct patients)')
print(T.round(3).to_string(index=False))

print('\nWhere do RULE errors go? (truth -> decision even with reference measurements)')
mr = typ == 'RULE'
print(pd.crosstab(pd.Series(np.array(PRIMARY)[yv[mr]], name='truth'), pd.Series(np.array(PRIMARY)[ref[mr]], name='ref decision')).to_string())
print('\nMedian reference LV_EF / LV_ED by class: ACDC vs M&Ms (is the population different?)')
iE, iV = FEATS.index('LV_EF'), FEATS.index('LV_ED')
for c in PRIMARY:
    a = classes[y_acdc] == c; b = R.label.values == c
    print(f'  {c}: LV_EF {np.median(A_gt[a, iE]):5.1f} vs {np.median(Cg[b, iE]):5.1f} | '
          f'LV_ED {np.median(A_gt[a, iV]):6.1f} vs {np.median(Cg[b, iV]):6.1f} mL')
print('\nMain source of doubt among SEG errors:'); print(pd.Series(doubt[typ == 'SEG']).value_counts().head(5).to_string())

pd.DataFrame(dict(code=R.code, vendor=vend, label=R.label, pred=np.array(PRIMARY)[pred], ref_decision=np.array(PRIMARY)[ref],
                  type=typ, flip_max=fmax, flag=flag, doubt=doubt)).to_csv(f'{OUT}/error_types.csv', index=False)
T.to_csv(f'{OUT}/flag_by_error_type.csv', index=False)
print('\nSaved to', OUT)
