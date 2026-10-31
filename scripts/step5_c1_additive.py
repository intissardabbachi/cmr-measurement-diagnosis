# =============================================================================
# STEP 5 — C1 (light): additive, auditable decision
#   logit(patient) = concept part  +  residual part
#     concept part : linear model on the 21 physical concepts (as step 2b)
#     residual part: linear model on frozen CineMA transformer features (PCA),
#                    fitted on top of the concept logits (offset), L2-penalised.
#   The penalty is chosen by inner CV over a grid that INCLUDES "no residual"
#   (lambda = inf), so the residual is used only when it helps on held-out data.
# Every patient gets an exact split of the evidence: concept share vs residual share.
#
# Leakage rules: per outer fold, PCA, scaler, both models and lambda are fitted on
# the training patients only. Concept offsets for training patients are cross-fitted
# (inner 5-fold) so the residual never sees in-sample concept logits.
# Test patients use PREDICTED concepts (step 3) and their CineMA features.
# CPU, about a minute.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
CINEMA_FEATS = os.environ.get('CINEMA_FEATS', f'{DRIVE}/cinema_features.csv')   # written by step_cinema.py (MODE=frozen)
OUT = f'{DRIVE}/step5_c1_additive'
os.makedirs(OUT, exist_ok=True)
SEED, B, NPC = 42, 2000, 32
LAMBDAS = [np.inf, 100.0, 30.0, 10.0, 3.0, 1.0]          # inf = no residual

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
R_all = pd.read_csv(CINEMA_FEATS).set_index('patient').loc[order].values
print(f'{n} patients | {len(FEATS)} concepts | {R_all.shape[1]} transformer features')

def concept_model():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                        {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')

def fit_residual(Z, Lc, yy, lam, w):
    """multinomial residual V (d x K) on top of fixed concept logits Lc; class weights w."""
    if np.isinf(lam):
        return np.zeros((Z.shape[1], K))
    Y = np.eye(K)[yy]
    sw = w[yy] / w[yy].sum()
    def obj(v):
        V = v.reshape(Z.shape[1], K)
        L = Lc + Z @ V
        loss = -(sw * (Y * log_softmax(L, 1)).sum(1)).sum() + lam / len(yy) * (V ** 2).sum()
        G = Z.T @ ((softmax(L, 1) - Y) * sw[:, None]) + 2 * lam / len(yy) * V
        return loss, G.ravel()
    r = minimize(obj, np.zeros(Z.shape[1] * K), jac=True, method='L-BFGS-B', options={'maxiter': 2000})
    return r.x.reshape(Z.shape[1], K)

def crossfit_concept_logits(idx):
    """concept logits for training patients, each predicted by a model that did not see it."""
    Lc = np.zeros((len(idx), K))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y[idx]):
        m = concept_model().fit(C_gt[idx[a]], y[idx[a]])
        Lc[b] = m.decision_function(C_gt[idx[b]])
    return Lc

P = np.zeros((n, K)); Lc_te_all = np.zeros((n, K)); Lr_te_all = np.zeros((n, K)); picks = []
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    w = len(tr) / (K * np.bincount(y[tr], minlength=K))
    # residual features: scaler + PCA on training patients only
    prep = make_pipeline(StandardScaler(), PCA(n_components=NPC, random_state=SEED)).fit(R_all[tr])
    Z_tr, Z_te = prep.transform(R_all[tr]), prep.transform(R_all[te])
    Z_tr, Z_te = Z_tr / Z_tr.std(0), Z_te / Z_tr.std(0)
    Lc_tr = crossfit_concept_logits(tr)

    # inner CV for lambda (includes 'no residual')
    scores = {}
    for lam in LAMBDAS:
        s = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y[tr]):
            V = fit_residual(Z_tr[a], Lc_tr[a], y[tr][a], lam, w)
            s.append(-np.mean(log_softmax(Lc_tr[b] + Z_tr[b] @ V, 1)[np.arange(len(b)), y[tr][b]]))   # NLL
        scores[lam] = np.mean(s)
    lam = min(scores, key=scores.get)

    cm = concept_model().fit(C_gt[tr], y[tr])
    V = fit_residual(Z_tr, Lc_tr, y[tr], lam, w)
    Lc_te = cm.decision_function(C_pr[te])                # predicted concepts of held-out patients
    Lr_te = Z_te @ V
    P[te] = softmax(Lc_te + Lr_te, 1)
    Lc_te_all[te], Lr_te_all[te] = Lc_te, Lr_te
    picks.append(dict(fold=k, lam=lam, **{f'nll_lam_{l}': v for l, v in scores.items()}))
    print(f'fold {k}: lambda = {lam} (inner NLL ' + ', '.join(f'{l}: {v:.3f}' for l, v in scores.items()) + ')')

# exact evidence split for the predicted class (centred logits)
pred = P.argmax(1)
c_part = Lc_te_all - Lc_te_all.mean(1, keepdims=True)
r_part = Lr_te_all - Lr_te_all.mean(1, keepdims=True)
cs, rs = np.abs(c_part[np.arange(n), pred]), np.abs(r_part[np.arange(n), pred])
share = cs / np.maximum(cs + rs, 1e-12)
ev = pd.DataFrame(dict(patient=order, label=folds.label.values, pred=classes[pred], concept_share=share))
print('\nConcept share of the evidence for the predicted class (1.0 = measurements only):')
print(ev.groupby('label').concept_share.describe()[['mean', 'min', '50%']].round(2).to_string())

# metrics + paired comparisons (same as previous steps)
Pd = {'additive C1 (ours)': P}
for name, path in (('concepts only, 21 (step 2b)', f'{DRIVE}/step2b_concepts21/oof_probs_concept21_LR_pred.npy'),
                   ('challenge RF (step 3)', f'{DRIVE}/step3_challenge_baseline/oof_probs_challenge_RF_pred.npy'),
                   ('CineMA frozen probe', f'{DRIVE}/cinema_oof_probs.npy')):
    if os.path.exists(path):
        Pd[name] = np.load(path)

def ece(Pm, yy, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == yy).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def metrics(Pm, i):
    return (balanced_accuracy_score(y[i], Pm[i].argmax(1)),
            roc_auc_score(y[i], Pm[i], multi_class='ovr', average='macro', labels=range(K)), ece(Pm[i], y[i]))

rng = np.random.default_rng(SEED)
boots = [i for i in (rng.integers(0, n, n) for _ in range(B)) if len(np.unique(y[i])) == K]
rows = []
for name, Pm in Pd.items():
    pt, bs = metrics(Pm, np.arange(n)), np.array([metrics(Pm, i) for i in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    rows.append(dict(model=name, bal_acc=pt[0], bal_lo=lo[0], bal_hi=hi[0], AUC=pt[1], AUC_lo=lo[1], AUC_hi=hi[1],
                     ECE=pt[2], macro_F1=f1_score(y, Pm.argmax(1), average='macro')))
res = pd.DataFrame(rows)
print('\n', res.round(3).to_string(index=False))
for ref in list(Pd)[1:3]:
    d = np.array([np.subtract(metrics(Pd['additive C1 (ours)'], i), metrics(Pd[ref], i)) for i in boots])
    print(f'\nPaired difference: additive C1 minus {ref}')
    for j, m in enumerate(['bal_acc', 'AUC', 'ECE']):
        print(f'  {m}: {d[:, j].mean():+.3f}  95% CI [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]')
print('\nConfusion matrix (rows = truth):', [str(c) for c in classes])
print(confusion_matrix(y, pred))

res.to_csv(f'{OUT}/step5_results.csv', index=False)
ev.to_csv(f'{OUT}/evidence_share.csv', index=False)
pd.DataFrame(picks).to_csv(f'{OUT}/model_selection.csv', index=False)
np.save(f'{OUT}/oof_probs_additive.npy', P)
print('\nSaved to', OUT)
