# =============================================================================
# STEP 7 — Measurement-error-aware diagnosis + conformal prediction sets
#
# Idea: the concept model is trained on reference measurements, but at test time it
# receives PREDICTED measurements (from segmentation) that carry an error we can
# estimate. Because the head is linear in the (standardised) concepts:
#     z = W s(c) + b,   c_true = c_pred - bias + e,  e ~ N(0, Sigma)
#  => logits are exactly Gaussian:  z ~ N(W s(c_pred - bias) + b,  W Sigma_s W^T)
# Probabilities = E[softmax(z)] (Monte Carlo over the exact Gaussian).
# Then split-conformal sets (CV+ style) give a coverage guarantee, and each patient
# gets an exact decomposition of WHICH measurement creates the doubt.
#
# Leakage rules (per outer fold): scaler, LR, C, bias, Sigma (Ledoit-Wolf) and the
# conformal threshold are estimated from TRAINING patients only. Calibration scores
# come from inner cross-fitting on training patients. Test patients: predicted concepts.
# Needs: folds.csv, step3_challenge_baseline/rich_features.csv. CPU, a few minutes.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
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
OUT = f'{DRIVE}/step7_eiv_conformal'
os.makedirs(OUT, exist_ok=True)
SEED, ALPHA, MC = 42, 0.10, 4000
rng = np.random.default_rng(SEED)

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
D = len(FEATS)
print(f'{n} patients | {D} concepts | alpha = {ALPHA}')

def fit_head(idx):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                     {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
    g.fit(C_gt[idx], y[idx])
    sc, lr = g.best_estimator_.named_steps['standardscaler'], g.best_estimator_.named_steps['logisticregression']
    return sc, lr.coef_, lr.intercept_

def error_model(idx):
    """bias and covariance of (pred - gt) measurement error, from these patients only."""
    E = C_pr[idx] - C_gt[idx]
    return E.mean(0), LedoitWolf().fit(E).covariance_

def predict(sc, W, b, bias, Sig, idx, mode):
    """mode: plain | var (propagate variance) | full (bias correction + variance)."""
    c = C_pr[idx] - (bias if mode == 'full' else 0)
    mu = ((c - sc.mean_) / sc.scale_) @ W.T + b                       # (m, K)
    if mode == 'plain':
        return softmax(mu, 1), np.zeros((len(idx), K, K))
    Ws = W / sc.scale_                                                # logit per raw unit
    Sz = Ws @ Sig @ Ws.T                                              # exact logit covariance (K x K)
    L = np.linalg.cholesky(Sz + 1e-9 * np.eye(K))
    eps = rng.standard_normal((MC, K)) @ L.T
    P = np.stack([softmax(m + eps, 1).mean(0) for m in mu])
    return P, np.broadcast_to(Sz, (len(idx), K, K))

def score(P, yy):                                                     # LAC nonconformity
    return 1 - P[np.arange(len(yy)), yy]

def aps_score(P, yy):                                                 # APS (non-randomised)
    o = np.argsort(-P, 1); cs = np.cumsum(np.take_along_axis(P, o, 1), 1)
    r = np.argmax(o == yy[:, None], 1)
    return cs[np.arange(len(yy)), r]

def qhat(s):
    m = len(s); return np.quantile(s, min(1, np.ceil((m + 1) * (1 - ALPHA)) / m), method='higher')

MODES = ['plain', 'var', 'full']
P = {m: np.zeros((n, K)) for m in MODES}
S_lac = {m: np.zeros((n, K), bool) for m in MODES}
S_aps = {m: np.zeros((n, K), bool) for m in MODES}
expl = []
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    # calibration scores: inner cross-fitting on training patients
    cal = {m: [np.zeros(len(tr)), np.zeros(len(tr))] for m in MODES}
    for a, bb in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y[tr]):
        sc, W, b0 = fit_head(tr[a]); bias, Sig = error_model(tr[a])
        for m in MODES:
            Pc, _ = predict(sc, W, b0, bias, Sig, tr[bb], m)
            cal[m][0][bb], cal[m][1][bb] = score(Pc, y[tr[bb]]), aps_score(Pc, y[tr[bb]])
    sc, W, b0 = fit_head(tr); bias, Sig = error_model(tr)
    for m in MODES:
        Pt, _ = predict(sc, W, b0, bias, Sig, te, m)
        P[m][te] = Pt
        q1, q2 = qhat(cal[m][0]), qhat(cal[m][1])
        S_lac[m][te] = (1 - Pt) <= q1
        o = np.argsort(-Pt, 1); cs = np.cumsum(np.take_along_axis(Pt, o, 1), 1)
        keep = np.zeros_like(Pt, bool)
        for i in range(len(te)):
            j = np.searchsorted(cs[i], q2 - 1e-12) + 1                # smallest prefix reaching q2
            keep[i, o[i, :min(j, K)]] = True
        S_aps[m][te] = keep
    # explanation: which measurement creates the doubt between the top-2 classes
    Ws = W / sc.scale_
    for t in te:
        Pt = P['full'][t]; a1, a2 = np.argsort(-Pt)[:2]
        d = Ws[a1] - Ws[a2]
        var = d @ Sig @ d
        margin = (((C_pr[t] - bias - sc.mean_) / sc.scale_) @ (W[a1] - W[a2])) + b0[a1] - b0[a2]
        # patient-specific: how many typical errors (SD) must measurement j be off to flip top1 -> top2
        flip = np.abs(margin / np.where(np.abs(d) > 1e-12, d, np.nan)) / np.sqrt(np.diag(Sig))
        top = np.argsort(np.nan_to_num(flip, nan=np.inf))[:3]
        expl.append(dict(patient=order[t], label=classes[y[t]], top1=classes[a1], top2=classes[a2],
                         margin=margin, margin_sd=np.sqrt(var), z=margin / np.sqrt(var),
                         **{f'doubt{r+1}': FEATS[j] for r, j in enumerate(top)},
                         **{f'doubt{r+1}_flipSD': flip[j] for r, j in enumerate(top)}))
    print(f'fold {k} done')

def ece(Pm, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

rows = []
for m in MODES:
    for nm, S in (('LAC', S_lac[m]), ('APS', S_aps[m])):
        cov = S[np.arange(n), y]
        rows.append(dict(model=m, sets=nm, bal_acc=balanced_accuracy_score(y, P[m].argmax(1)),
                         AUC=roc_auc_score(y, P[m], multi_class='ovr'), ECE=ece(P[m]),
                         NLL=-np.mean(np.log(np.clip(P[m][np.arange(n), y], 1e-12, 1))),
                         coverage=cov.mean(), mean_size=S.sum(1).mean(), singleton=(S.sum(1) == 1).mean(),
                         **{f'cov_{c}': cov[y == i].mean() for i, c in enumerate(classes)}))
res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(f'\nTarget coverage = {1 - ALPHA:.2f}')
print(res.round(3).to_string(index=False))

ex = pd.DataFrame(expl)
print('\nMost frequent source of doubt (top-1 measurement), full model:')
print(ex.doubt1.value_counts().head(8).to_string())
wrong = ex[ex.label != ex.top1]
print(f'\nMisdiagnosed patients ({len(wrong)}): was the truth the 2nd choice? {np.mean(wrong.label == wrong.top2) * 100:.0f}%')
print(wrong[['patient', 'label', 'top1', 'top2', 'z', 'doubt1', 'doubt1_flipSD']].round(2).to_string(index=False))
print('\n|z| (margin / its SD) for correct vs wrong:',
      f'{ex[ex.label == ex.top1].z.abs().median():.2f} vs {wrong.z.abs().median():.2f}')

res.to_csv(f'{OUT}/step7_results.csv', index=False)
ex.to_csv(f'{OUT}/doubt_explanations.csv', index=False)
for m in MODES:
    np.save(f'{OUT}/oof_probs_{m}.npy', P[m]); np.save(f'{OUT}/sets_LAC_{m}.npy', S_lac[m])
print('\nSaved to', OUT)
