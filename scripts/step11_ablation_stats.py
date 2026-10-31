# =============================================================================
# STEP 11 — Ablations / sensitivity analyses + paired tests (CPU, ~5-10 min)
# Main choices stay locked (21 measurements, S = 20, flag threshold 0.20, kappa fitted).
# Everything below is reported as ablation / sensitivity analysis.
#   A. Measurement groups (ACDC, same protocol as step 2b): drop one group at a time
#   B. Number of stochastic samples S in {5, 10, 20} (ACDC flag + spread-error agreement)
#   C. Flag threshold in {0.10, 0.20, 0.30, 0.40} (ACDC and M&Ms after the ED/ES rule)
#   D. kappa = 1 vs kappa fitted (ACDC flag)
#   E. Paired tests: exact McNemar (correct / wrong per patient, Holm-corrected) and
#      Wilcoxon signed-rank on per-patient Brier score and NLL
# Needs outputs of steps 2b, 3, 8, 9c (and CineMA / step 4 probabilities if present).
# =============================================================================
import os, glob, numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
OUT = f'{DRIVE}/step11_ablation'
os.makedirs(OUT, exist_ok=True)
SEED, B, REP = 42, 2000, 20
rng = np.random.default_rng(SEED)
pd.set_option('display.width', 220)

folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
n, K = len(y), len(classes)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
C_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
C_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
CS = np.load(f'{DRIVE}/step8_patient_uncertainty/concept_samples.npz')['CS']
F = {c: j for j, c in enumerate(FEATS)}

def head(cols=None):
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                        {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')

def metrics(P, i):
    return np.array([balanced_accuracy_score(y[i], P[i].argmax(1)),
                     roc_auc_score(y[i], P[i], multi_class='ovr', labels=range(K))])
boots = [i for i in (rng.integers(0, n, n) for _ in range(B)) if len(np.unique(y[i])) == K]

# ------------------------------------------------ A. measurement groups
GROUPS = {'volumes (6)': [f'{a}_{p}' for a in ('RV', 'MYO', 'LV') for p in ('ED', 'ES')],
          'ejection fractions (2)': ['LV_EF', 'RV_EF'],
          'LV mass (1)': ['LV_mass'],
          'ratios (4)': ['RV_LV_ED', 'RV_LV_ES', 'MYO_LV_ED', 'MYO_LV_ES'],
          'wall thickness (8)': [c for c in FEATS if c.startswith('WT_')]}
def oof(cols):
    idx = [F[c] for c in cols]; P = np.zeros((n, K))
    for k in range(5):
        tr, te = f != k, f == k
        m = head().fit(C_gt[tr][:, idx], y[tr]); P[te] = m.predict_proba(C_pr[te][:, idx])
    return P
P_full = oof(FEATS)
bs_full = np.array([metrics(P_full, i) for i in boots])
rows = [dict(variant='all 21 measurements', n_feat=21, bal_acc=metrics(P_full, np.arange(n))[0],
             AUC=metrics(P_full, np.arange(n))[1], d_bal=0.0, lo=0.0, hi=0.0)]
for g, cols in GROUPS.items():
    keep = [c for c in FEATS if c not in cols]
    P = oof(keep); d = np.array([metrics(P, i) for i in boots]) - bs_full
    pt = metrics(P, np.arange(n))
    rows.append(dict(variant=f'without {g}', n_feat=len(keep), bal_acc=pt[0], AUC=pt[1],
                     d_bal=pt[0] - rows[0]['bal_acc'], lo=np.percentile(d[:, 0], 2.5), hi=np.percentile(d[:, 0], 97.5)))
    print(f'A: without {g} done')
TA = pd.DataFrame(rows)
print('\nA. Measurement-group ablation (ACDC, out-of-fold; d_bal = minus full, paired bootstrap 95% CI):')
print(TA.round(3).to_string(index=False))

# ------------------------------------------------ B, C, D. flag on ACDC
heads, wrong = {}, np.zeros(n, bool)
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    heads[k] = head().fit(C_gt[tr], y[tr]).best_estimator_
    wrong[te] = heads[k].predict(C_pr[te]) != y[te]
print(f'\nACDC errors of the main rule: {wrong.sum()}')

def flag_scores(samples, kappa_mode):
    """max flip fraction per patient; kappa fitted per fold on training patients (or 1)."""
    dv = samples - samples.mean(1, keepdims=True); sd = samples.std(1) + 1e-9
    fmax = np.zeros(n)
    for k in range(5):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        kap = 1.0 if kappa_mode == 'one' else float(np.median(np.abs(C_pr[tr] - C_gt[tr]) / sd[tr]) / 0.6745)
        m = heads[k]
        for t in te:
            base = m.predict(C_pr[[t]])[0]
            fl = []
            for j in range(len(FEATS)):
                X = np.repeat(C_pr[[t]], samples.shape[1], 0); X[:, j] += kap * dv[t, :, j]
                fl.append(np.mean(m.predict(X) != base))
            fmax[t] = max(fl)
    return fmax

def flag_summary(fmax, w, thr=0.20):
    fl = fmax >= thr
    return dict(AUROC=roc_auc_score(w, fmax), sens=fl[w].mean(), false_flag=fl[~w].mean())

conf_auc = roc_auc_score(wrong, 1 - np.load(f'{DRIVE}/step8_patient_uncertainty/oof_probs_plain.npy').max(1))
rows = []
for S in (5, 10, 20):
    reps = 1 if S == 20 else REP
    res = []
    for _ in range(reps):
        sub = np.stack([CS[i, rng.choice(20, S, replace=False)] for i in range(n)]) if S < 20 else CS
        fm = flag_scores(sub, 'fit')
        rho = np.median([spearmanr(sub.std(1)[:, F[c]], np.abs(C_pr[:, F[c]] - C_gt[:, F[c]]))[0]
                         for c in ('LV_EF', 'RV_EF', 'LV_ED', 'RV_ED', 'MYO_ED', 'RV_LV_ED', 'MYO_LV_ED')])
        res.append(dict(**flag_summary(fm, wrong), median_rho=rho))
    r = pd.DataFrame(res)
    rows.append(dict(S=S, repeats=reps, **{c: r[c].mean() for c in r}, AUROC_sd=r.AUROC.std() if reps > 1 else 0.0))
    print(f'B: S = {S} done')
TB = pd.DataFrame(rows)
print(f'\nB. Number of samples S (ACDC; flag threshold 0.20; confidence AUROC = {conf_auc:.3f}):')
print(TB.round(3).to_string(index=False))

fm20 = flag_scores(CS, 'fit'); fm1 = flag_scores(CS, 'one')
TD = pd.DataFrame([dict(kappa='fitted (main)', **flag_summary(fm20, wrong)), dict(kappa='1 (no inflation)', **flag_summary(fm1, wrong))])
print('\nD. Inflation factor (ACDC flag, S = 20):'); print(TD.round(3).to_string(index=False))

ext = pd.read_csv(f'{DRIVE}/step9_mnms_external/step9c_per_patient.csv')
w_ext = (ext.type != 'CORRECT').values; seg = (ext.type == 'SEG').values; ok = (ext.type == 'CORRECT').values
rows = []
for thr in (0.10, 0.20, 0.30, 0.40):
    a = flag_summary(fm20, wrong, thr); fe = ext.flip_max.values >= thr
    rows.append(dict(threshold=thr, ACDC_sens=a['sens'], ACDC_false_flag=a['false_flag'],
                     MnMs_sens_all=fe[w_ext].mean(), MnMs_sens_SEG=fe[seg].mean(), MnMs_false_flag=fe[ok].mean()))
TC = pd.DataFrame(rows)
print('\nC. Flag threshold (main = 0.20):'); print(TC.round(3).to_string(index=False))

# ------------------------------------------------ E. paired tests
P = {'logistic rule (main)': np.load(f'{DRIVE}/step2b_concepts21/oof_probs_concept21_LR_pred.npy'),
     'random forest': np.load(f'{DRIVE}/step3_challenge_baseline/oof_probs_challenge_RF_pred.npy')}
for name, pat in (('CineMA linear probe', f'{DRIVE}/cinema_oof_probs.npy'),
                  ('additive model', f'{DRIVE}/step5_c1_additive/oof_probs_additive.npy'),
                  ('per-patient uncertainty (pp_cal)', f'{DRIVE}/step8_patient_uncertainty/oof_probs_pp_cal.npy')):
    if os.path.exists(pat): P[name] = np.load(pat)
j4 = sorted(glob.glob(f'{DRIVE}/step4_joint/*oof*prob*.npy'))
if j4: P['joint model'] = np.load(j4[0]); print('joint model probabilities:', os.path.basename(j4[0]))

def brier(Pm): return ((Pm - np.eye(K)[y]) ** 2).sum(1)
def nll(Pm): return -np.log(np.clip(Pm[np.arange(n), y], 1e-12, 1))
ref = P['logistic rule (main)']; c_ref = ref.argmax(1) == y
rows = []
for name, Pm in P.items():
    if name == 'logistic rule (main)': continue
    c = Pm.argmax(1) == y
    b01, b10 = int((c_ref & ~c).sum()), int((~c_ref & c).sum())    # main right / other wrong, and reverse
    p_mc = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
    wb, wn = wilcoxon(brier(ref), brier(Pm)), wilcoxon(nll(ref), nll(Pm))
    rows.append(dict(comparison=f'main vs {name}', acc_main=c_ref.mean(), acc_other=c.mean(), main_only_right=b01,
                     other_only_right=b10, McNemar_p=p_mc, median_dBrier=np.median(brier(ref) - brier(Pm)),
                     Wilcoxon_Brier_p=wb.pvalue, median_dNLL=np.median(nll(ref) - nll(Pm)), Wilcoxon_NLL_p=wn.pvalue))
TE = pd.DataFrame(rows)
m = len(TE); o = np.argsort(TE.McNemar_p.values); adj = np.zeros(m)
for r_, i in enumerate(o):                                        # Holm step-down
    adj[i] = min(1.0, max(adj[o[r_ - 1]] if r_ else 0, (m - r_) * TE.McNemar_p.values[i]))
TE['McNemar_p_Holm'] = adj
print('\nE. Paired tests against the main rule (ACDC, n = 100). dBrier/dNLL = main minus other (negative = main better):')
print(TE.round(4).to_string(index=False))

for name, T in (('A_groups', TA), ('B_samples', TB), ('C_threshold', TC), ('D_kappa', TD), ('E_paired_tests', TE)):
    T.to_csv(f'{OUT}/{name}.csv', index=False)
print('\nSaved to', OUT)
