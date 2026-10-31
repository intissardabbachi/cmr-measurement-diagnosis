# =============================================================================
# STEP 6 — Feasibility: are the two heart laws a usable label-free signal?
#   Law 1 (myocardium incompressible): MYO_ES ≈ MYO_ED      -> r1 = MYO_ES/MYO_ED - 1
#   Law 2 (same blood out of both ventricles): SV_LV ≈ SV_RV -> r2 = SV_LV - SV_RV (mL)
# Question: is the violation caused by SEGMENTATION ERROR (law can guide adaptation)
#           or by the HEART itself (law would mislead)?
#   -> compare violations on reference masks (gt) vs predicted masks (pred),
#      and check if violation on pred tracks the pred-vs-gt measurement error.
# Needs only step3_challenge_baseline/rich_features.csv. CPU, seconds.
# =============================================================================
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
OUT = f'{DRIVE}/step6_feasibility'
os.makedirs(OUT, exist_ok=True)

folds = pd.read_csv(f'{DRIVE}/folds.csv')
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
G = rf[rf.source == 'gt'].set_index('patient').loc[folds.patient]
P = rf[rf.source == 'pred'].set_index('patient').loc[folds.patient]
lab = folds.label.values

def laws(D):
    r1 = 100 * (D.MYO_ES / D.MYO_ED - 1)                          # % change of myocardium volume
    r2 = (D.LV_ED - D.LV_ES) - (D.RV_ED - D.RV_ES)                # SV_LV - SV_RV (mL)
    return r1.values, r2.values

g1, g2 = laws(G)
p1, p2 = laws(P)

# 1) how well does each source obey the laws, per class
rows = []
for c in ['ALL'] + sorted(set(lab)):
    m = np.ones(len(lab), bool) if c == 'ALL' else lab == c
    rows.append(dict(cls=c, n=m.sum(),
                     MYO_gt_mean=g1[m].mean(), MYO_gt_absmed=np.median(abs(g1[m])),
                     MYO_pred_mean=p1[m].mean(), MYO_pred_absmed=np.median(abs(p1[m])),
                     SV_gt_mean=g2[m].mean(), SV_gt_absmed=np.median(abs(g2[m])),
                     SV_pred_mean=p2[m].mean(), SV_pred_absmed=np.median(abs(p2[m]))))
tab = pd.DataFrame(rows)
print('Law violations (MYO in %, SV in mL); absmed = median of |violation|')
print(tab.round(1).to_string(index=False))

# 2) does pred violate MORE than gt? (paired, per patient)
for name, a, b in (('MYO law', p1, g1), ('SV law', p2, g2)):
    s, pv = wilcoxon(abs(a), abs(b))
    print(f'\n{name}: |violation| pred > gt in {np.mean(abs(a) > abs(b)) * 100:.0f}% of patients, Wilcoxon p = {pv:.2g}')

# 3) the part of the pred violation that is NOT already in gt = segmentation-induced
#    does it track the actual measurement error? (this is what adaptation would fix)
e_myo = abs(P.MYO_ES.values - G.MYO_ES.values) + abs(P.MYO_ED.values - G.MYO_ED.values)
e_sv = abs((P.LV_ED - P.LV_ES).values - (G.LV_ED - G.LV_ES).values) + abs((P.RV_ED - P.RV_ES).values - (G.RV_ED - G.RV_ES).values)
e_ef = abs(P.LV_EF.values - G.LV_EF.values) + abs(P.RV_EF.values - G.RV_EF.values)
print('\nSpearman: |violation on pred|  vs  measurement error (pred vs gt)')
for name, v in (('MYO law', abs(p1)), ('SV law', abs(p2))):
    for en, e in (('MYO vol error', e_myo), ('SV error', e_sv), ('EF error', e_ef)):
        r, pv = spearmanr(v, e)
        print(f'  {name:8s} vs {en:14s}: rho = {r:+.2f}  p = {pv:.2g}')

# 4) share of pred violation explained by segmentation (variance decomposition)
for name, a, b in (('MYO law', p1, g1), ('SV law', p2, g2)):
    print(f'\n{name}: var(gt) = {b.var():.1f}, var(pred) = {a.var():.1f}, '
          f'var(pred - gt) = {(a - b).var():.1f}  -> seg-induced share ≈ {(a - b).var() / a.var() * 100:.0f}%')

tab.to_csv(f'{OUT}/law_violations_by_class.csv', index=False)
pd.DataFrame(dict(patient=folds.patient, label=lab, myo_gt=g1, myo_pred=p1, sv_gt=g2, sv_pred=p2,
                  err_myo=e_myo, err_sv=e_sv, err_ef=e_ef)).to_csv(f'{OUT}/per_patient.csv', index=False)
print('\nSaved to', OUT)
