# =============================================================================
# STEP 12 — Publication figures (CPU, < 1 min). Uses saved outputs only.
#   Fig 2  ROC: flip flag vs softmax confidence (ACDC) + flag by error type (M&Ms-1)
#   Fig 3  per-patient sample SD vs real |error| (LV EF, RV EDV; ACDC and M&Ms-1)
#   Fig 4  (a) error attribution per vendor  (b) reference LV EF by class, ACDC vs M&Ms-1
#   Fig 5  few-shot recalibration: balanced accuracy vs k per vendor
# (Fig 1 = pipeline diagram, already in the manuscript.)
# Colours: Okabe-Ito subset (colour-blind safe, validated) + grey; every series also
# differs by marker or line style, so the figures survive grey-scale printing.
# Output: step12_figures/fig{2..5}.pdf and .png (300 dpi)
# =============================================================================
import os, io, glob, zipfile, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
OUT = f'{DRIVE}/step12_figures'
os.makedirs(OUT, exist_ok=True)
BLUE, VERM, GREEN, GREY, INK = '#0072B2', '#D55E00', '#009E73', '#8C8C8C', '#222222'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'axes.edgecolor': '#444444', 'axes.labelcolor': INK,
                     'xtick.color': '#444444', 'ytick.color': '#444444', 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}')
    plt.close(fig); print('saved', name)
def panel(ax, letter):
    t = ax.get_title(); ax.set_title('')
    ax.set_title(r'$\bf{' + letter + '}$   ' + t, loc='left')

# ---------------- shared data
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
F = {c: j for j, c in enumerate(FEATS)}
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
A_cs = np.load(f'{DRIVE}/step8_patient_uncertainty/concept_samples.npz')['CS']

def swap_phases(X):                                              # same ED/ES rule as step 9c
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
R = pd.DataFrame(list(np.load(f'{DRIVE}/step9_mnms_external/mnms_concepts.npz', allow_pickle=True)['res']))
M_gt, M_pr, M_cs = np.stack(list(R['gt'])), np.stack(list(R['pred'])), np.stack(list(R['samples']))
sw = M_pr[:, F['LV_ED']] < M_pr[:, F['LV_ES']]
M_gt[sw], M_pr[sw], M_cs[sw] = swap_phases(M_gt[sw]), swap_phases(M_pr[sw]), swap_phases(M_cs[sw])
ext = pd.read_csv(f'{DRIVE}/step9_mnms_external/step9c_per_patient.csv')
R['centre'] = ext.set_index('code').loc[R.code, 'centre'].values

# ---------------- Fig 2: ROC
dp = pd.read_csv(f'{DRIVE}/step8_patient_uncertainty/doubt_per_patient.csv').set_index('patient').loc[order]
wrong = (dp.label != dp.pred).values
flip = dp.doubt1_flip.values
conf = 1 - np.load(f'{DRIVE}/step8_patient_uncertainty/oof_probs_plain.npy').max(1)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.8))
ax = axs[0]
for s, c, ls, lab in ((flip, BLUE, '-', 'Flip flag'), (conf, GREY, '--', 'Softmax confidence')):
    fpr, tpr, _ = roc_curve(wrong, s)
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1.6, drawstyle='steps-post', label=f'{lab} (AUROC {roc_auc_score(wrong, s):.2f})')
thr = flip >= 0.20
ax.plot(thr[~wrong].mean(), thr[wrong].mean(), 'o', ms=6, mfc='white', mec=BLUE, mew=1.4, label='Flag threshold 0.20')
ax.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.8, zorder=0)
ax.set(xlabel='False-positive rate (correct patients flagged)', ylabel='True-positive rate (errors flagged)',
       title=f'ACDC: {wrong.sum()} errors, {(~wrong).sum()} correct', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
ax.set_aspect('equal'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2)); panel(ax, 'a')
ax = axs[1]
ok = (ext.type == 'CORRECT').values
for t, c, ls, lab in (('SEG', BLUE, '-', 'Segmentation-driven errors'), ('RULE', VERM, '-.', 'Rule-driven errors')):
    m = (ext.type == t).values; sel = m | ok
    fpr, tpr, _ = roc_curve(m[sel], ext.flip_max.values[sel])
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1.6, drawstyle='steps-post',
            label=f'{lab}, n = {m.sum()} (AUROC {roc_auc_score(m[sel], ext.flip_max.values[sel]):.2f})')
ax.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.8, zorder=0)
ax.set(xlabel='False-positive rate (correct patients flagged)', ylabel='True-positive rate',
       title=f'M&Ms-1: flag by error type ({ok.sum()} correct)', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
ax.set_aspect('equal'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2)); panel(ax, 'b')
fig.tight_layout(w_pad=2.5); save(fig, 'fig2_flag_roc')

# ---------------- Fig 3: uncertainty vs error
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
for r_, (feat, unit, name) in enumerate((('LV_EF', 'percentage points', 'LV ejection fraction'), ('RV_ED', 'mL', 'RV end-diastolic volume'))):
    for c_, (lab, gt, pr, cs, col, mk) in enumerate((('ACDC (n = 100)', A_gt, A_pr, A_cs, BLUE, 'o'),
                                                     (f'M&Ms-1 (n = {len(R)})', M_gt, M_pr, M_cs, VERM, 's'))):
        ax = axs[r_, c_]; j = F[feat]
        sd, err = cs[:, :, j].std(1), np.abs(pr[:, j] - gt[:, j])
        rho, p = spearmanr(sd, err)
        fx, fy = np.percentile(sd[sd > 0], 1), np.percentile(err[err > 0], 1)
        ax.scatter(np.maximum(sd, fx), np.maximum(err, fy), s=10, color=col, marker=mk, alpha=0.55, linewidths=0)
        ax.set_xscale('log'); ax.set_yscale('log')
        for a_ in (ax.xaxis, ax.yaxis):
            a_.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            a_.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a_.set_minor_formatter(NullFormatter())
        ax.set(xlabel=f'Per-patient sample SD ({unit})', ylabel=f'|Predicted − reference| ({unit})', title=f'{name}, {lab}')
        ax.text(0.03, 0.95, f'Spearman ρ = {rho:.2f}, p = {p:.1g}', transform=ax.transAxes, va='top', fontsize=7, color=INK, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC', lw=0.5))
        panel(ax, 'abcd'[2 * r_ + c_])
fig.tight_layout(h_pad=2.0, w_pad=2.5); save(fig, 'fig3_uncertainty_vs_error')

# ---------------- Fig 4: error attribution + population shift
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1, 1.1]})
ax = axs[0]
vend = sorted(ext.vendor.unique(), key=lambda v: -(ext.vendor == v).sum())
cnt = pd.crosstab(ext.vendor, ext.type).reindex(vend).fillna(0)
for k in ('CORRECT', 'SEG', 'RULE'):
    if k not in cnt: cnt[k] = 0
prop = cnt[['CORRECT', 'SEG', 'RULE']].div(cnt.sum(axis=1), axis=0)
left = np.zeros(len(vend))
for k, col, hatch, lab in (('CORRECT', '#D9D9D9', None, 'Correct'), ('SEG', BLUE, None, 'Segmentation-driven error'),
                           ('RULE', VERM, '////', 'Rule-driven error')):
    ax.barh(range(len(vend)), prop[k], left=left, color=col, hatch=hatch, edgecolor='white', linewidth=1.0, height=0.62, label=lab)
    left += prop[k].values
ax.set_yticks(range(len(vend)), [f'{v} (n = {int(cnt.loc[v].sum())})' for v in vend]); ax.invert_yaxis()
ax.set(xlabel='Share of patients', xlim=(0, 1), title='Error source by vendor')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=1); panel(ax, 'a')
ax = axs[1]
rng = np.random.default_rng(0)
labs = ['DCM', 'HCM', 'NOR']
for i, cl in enumerate(labs):
    a = A_gt[classes[y] == cl, F['LV_EF']]
    ax.scatter(i - 0.18 + rng.uniform(-0.07, 0.07, len(a)), a, s=9, color=BLUE, marker='o', alpha=0.7, linewidths=0,
               label='ACDC' if i == 0 else None)
    m = (R.label == cl).values
    for cen3, mfc in ((False, VERM), (True, 'white')):
        mm = m & ((R.centre.values == 3) == cen3)
        b = M_gt[mm, F['LV_EF']]
        if len(b):
            ax.scatter(i + 0.18 + rng.uniform(-0.07, 0.07, len(b)), b, s=9, marker='s', facecolors=mfc, edgecolors=VERM,
                       linewidths=0.6, alpha=0.8, label=('M&Ms-1' if not cen3 else 'M&Ms-1, centre 3') if i == 0 else None)
    for x0, v in ((i - 0.18, a), (i + 0.18, M_gt[m, F['LV_EF']])):
        ax.plot([x0 - 0.1, x0 + 0.1], [np.median(v)] * 2, color=INK, lw=1.4)
ax.set_xticks(range(3), labs); ax.set(ylabel='Reference LV ejection fraction (%)', title='Same label, different patients')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2); panel(ax, 'b')
fig.tight_layout(w_pad=4.0); save(fig, 'fig4_error_attribution')

# ---------------- Fig 5: few-shot
T = pd.read_csv(f'{DRIVE}/step9_mnms_external/step10_fewshot.csv')
vs = [v for v in ('Siemens', 'Philips', 'GE', 'Canon') if v in set(T.vendor)]
fig, axs = plt.subplots(1, len(vs), figsize=(6.6, 2.4), sharey=True)
for ax, v in zip(np.atleast_1d(axs), vs):
    t = T[T.vendor == v]
    a0 = t[(t.method == 'A0 locked') & (t.k == 0)].bal_acc.values[0]
    ax.axhline(a0, color=GREY, ls='--', lw=1.0, label='A0 locked')
    for meth, col, mk, ls, lab in (('A1 intercept-only', BLUE, 'o', '-', 'A1 offsets only'), ('A2 joint refit', VERM, 's', '-.', 'A2 full refit')):
        s = t[t.method == meth].sort_values('k')
        ax.fill_between(s.k, s.bal_acc - s.sd, s.bal_acc + s.sd, color=col, alpha=0.12, linewidth=0)
        ax.plot(s.k, s.bal_acc, color=col, marker=mk, ms=4, ls=ls, lw=1.4, label=lab)
    ax.set(title=f'{v} (n = {int(t[t.k == 0].n_test.values[0])})', xticks=sorted(t[t.k > 0].k.unique()))
axs[0].set_ylabel('Balanced accuracy (held-out)')
fig.supxlabel('Labelled patients per class from the new vendor (k)', fontsize=8, y=0.02)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=1.2, rect=(0, 0.04, 1, 1)); save(fig, 'fig5_fewshot')
print('All figures saved to', OUT)
