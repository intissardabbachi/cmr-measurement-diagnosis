# =============================================================================
# STEP 2b — Concept model on the full set of 21 physical concepts
# Decision (recorded before any external test): the concept set is extended from 7
# to the same 21 physical quantities used by the strongest same-family baseline
# (step 3), so that both models receive the SAME information. Only the head differs:
#   ours  = multinomial logistic regression (one readable weight per concept and class)
#   step3 = random forest
# Same protocol: fitted on reference concepts of training patients (C by inner CV),
# applied to predicted concepts of held-out patients. CPU, seconds.
# Needs: step3_challenge_baseline/rich_features.csv and oof_probs_challenge_RF_pred.npy
# =============================================================================
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S3 = f'{DRIVE}/step3_challenge_baseline'
OUT = f'{DRIVE}/step2b_concepts21'
os.makedirs(OUT, exist_ok=True)
SEED, B = 42, 2000

folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
rf = pd.read_csv(f'{S3}/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
R = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
Q = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
print(len(FEATS), 'concepts')

P = {'concept21_LR_pred (ours)': np.zeros((len(y), len(classes))),
     'concept21_LR_GT_upper': np.zeros((len(y), len(classes)))}
coefs = []
for k in range(5):
    tr, te = f != k, f == k
    grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                        {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
    grid.fit(R[tr], y[tr])
    P['concept21_LR_pred (ours)'][te] = grid.predict_proba(Q[te])
    P['concept21_LR_GT_upper'][te] = grid.predict_proba(R[te])
    lr = grid.best_estimator_.named_steps['logisticregression']
    coefs += [dict(fold=k, cls=str(classes[i]), **dict(zip(FEATS, lr.coef_[i]))) for i in range(len(classes))]
    print(f'fold {k}: C = {grid.best_params_["logisticregression__C"]}')

others = {'challenge_RF_pred': f'{S3}/oof_probs_challenge_RF_pred.npy',
          'concept7_LR_pred (previous)': f'{DRIVE}/step1_unet/oof_probs_concept_pred_soft.npy',
          'cinema_frozen_probe': f'{DRIVE}/cinema_oof_probs.npy'}
for name, path in others.items():
    if os.path.exists(path):
        P[name] = np.load(path)

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
            roc_auc_score(y[i], Pm[i], multi_class='ovr', average='macro', labels=range(len(classes))), ece(Pm[i], y[i]))

rng = np.random.default_rng(SEED)
boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if len(np.unique(y[i])) == len(classes)]
rows = []
for name, Pm in P.items():
    pt, bs = metrics(Pm, np.arange(len(y))), np.array([metrics(Pm, i) for i in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    rows.append(dict(model=name, bal_acc=pt[0], bal_lo=lo[0], bal_hi=hi[0], AUC=pt[1], AUC_lo=lo[1], AUC_hi=hi[1],
                     ECE=pt[2], macro_F1=f1_score(y, Pm.argmax(1), average='macro')))
res = pd.DataFrame(rows)
print('\n', res.round(3).to_string(index=False))

ours = 'concept21_LR_pred (ours)'
for ref in ('challenge_RF_pred', 'concept7_LR_pred (previous)'):
    if ref in P:
        d = np.array([np.subtract(metrics(P[ours], i)[:2], metrics(P[ref], i)[:2]) for i in boots])
        print(f'\nPaired difference: {ours} minus {ref}')
        for j, m in enumerate(['bal_acc', 'AUC']):
            print(f'  {m}: {d[:, j].mean():+.3f}  95% CI [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]')

print('\nConfusion matrix, ours (rows = truth):', [str(c) for c in classes])
print(confusion_matrix(y, P[ours].argmax(1)))
res.to_csv(f'{OUT}/step2b_results.csv', index=False)
pd.DataFrame(coefs).to_csv(f'{OUT}/step2b_coefficients.csv', index=False)
np.save(f'{OUT}/oof_probs_concept21_LR_pred.npy', P[ours])
print('\nSaved to', OUT)
