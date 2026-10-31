# =============================================================================
# CineMA baselines under the SAME conditions as our model
# (same 100 ACDC patients, same folds.csv, same inner-validation patients,
#  same metrics, predictions saved in folds.csv order for paired comparison)
#
#   MODE=frozen    pretrained encoder frozen + logistic regression (C by inner CV)
#   MODE=finetune  full fine-tuning with the authors' training components
#                  (ConvViT.from_pretrained, layer-wise lr decay, warmup + cosine,
#                  AdamW, label smoothing 0.1, grad clipping 5); checkpoint chosen
#                  on the inner-validation patients (macro AUC, as in their code)
#
# Weights: ONLY pretrained/cinema.safetensors. The ACDC fine-tuned checkpoints
# released by the authors are never used (they saw our test patients).
#
# Needs the CineMA repo installed (Kaggle/Colab cell):
#   !git clone -q https://github.com/mathpluscode/CineMA.git && pip -q install -e CineMA
# Env: MODE (frozen), EPOCHS (100), BATCH (4), LR (1e-3 * BATCH / 64, linear scaling)
# =============================================================================
import os, glob, time, copy, zipfile
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from omegaconf import OmegaConf
import cinema
from cinema import ConvViT

MODE = os.environ.get('MODE', 'frozen')
KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if KAGGLE:
    DRIVE = '/kaggle/working'
    ROOT = os.path.dirname(os.path.dirname(sorted(glob.glob('/kaggle/input/**/patient001/Info.cfg', recursive=True))[0]))
    FOLDS = sorted(glob.glob('/kaggle/input/**/folds.csv', recursive=True))[0]
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except (ImportError, NotImplementedError):
        pass
    DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
    ROOT = '/content/acdc/training'
    FOLDS = f'{DRIVE}/folds.csv'
    if not os.path.isdir(ROOT):
        zipfile.ZipFile('/content/drive/MyDrive/ACDC_calibration_article/training-20261021T163705Z-1-001.zip').extractall('/content/acdc')
ROI = os.environ.get('ROI', '0') == '1'      # crop around the heart (centre from out-of-fold U-Net masks)
OUT = f"{DRIVE}/step_cinema_{MODE}{'_roi' if ROI else ''}{os.environ.get('RUN_TAG', '')}"
os.makedirs(OUT, exist_ok=True)
SEED, PATCH = 42, (192, 192, 16)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'MODE {MODE} | device {dev} | data {ROOT} | out {OUT}')

# ------------------------------------------------ preprocessing (as in our frozen run)
def read_cfg(p):
    return {k.strip(): v.strip() for k, v in (l.split(':', 1) for l in open(p) if ':' in l)}

def nii(base):
    for e in ('.nii.gz', '.nii'):
        if os.path.exists(base + e):
            return base + e
    raise FileNotFoundError(base)

def fit(v, shape):
    out = np.zeros(shape, np.float32)
    src, dst = [], []
    for a, t in zip(v.shape, shape):
        n = min(a, t)
        src.append(slice((a - n) // 2, (a - n) // 2 + n))
        dst.append(slice((t - n) // 2, (t - n) // 2 + n))
    out[tuple(dst)] = v[tuple(src)]
    return out

CENTERS = None
if ROI:                                      # heart_centers.csv: patient, cx, cy, cz in native voxel indices (ED)
    cpath = sorted(glob.glob('/kaggle/input/**/heart_centers.csv', recursive=True)) if KAGGLE else [f'{DRIVE}/heart_centers.csv']
    CENTERS = pd.read_csv(cpath[0]).set_index('patient')

def fit_at(v, shape, centre):
    """crop/pad v to shape so that `centre` (voxel coords) lands in the middle."""
    out = np.zeros(shape, np.float32)
    src, dst = [], []
    for a, t, c in zip(v.shape, shape, centre):
        s0 = int(round(c)) - t // 2
        lo, hi = max(s0, 0), min(s0 + t, a)
        src.append(slice(lo, hi)); dst.append(slice(lo - s0, lo - s0 + (hi - lo)))
    out[tuple(dst)] = v[tuple(src)]
    return out

def load(pid):
    d = f'{ROOT}/{pid}'
    info = read_cfg(f'{d}/Info.cfg')
    ch = []
    for fr in (int(info['ED']), int(info['ES'])):
        img = nib.load(nii(f'{d}/{pid}_frame{fr:02d}'))
        sx, sy, sz = img.header.get_zooms()[:3]
        r = zoom(img.get_fdata().astype(np.float32), (sx, sy, sz / 10.0), order=1)            # 1x1x10 mm
        if ROI:
            c = CENTERS.loc[pid]
            v = fit_at(r, PATCH, (c.cx * sx, c.cy * sy, c.cz * sz / 10.0))
        else:
            v = fit(r, PATCH)
        ch.append((v - v.min()) / (v.max() - v.min() + 1e-8))
    return np.stack(ch)                                        # (2, 192, 192, 16): ED, ES

folds = pd.read_csv(FOLDS)
classes = sorted(folds.label.unique())                         # DCM, HCM, MINF, NOR, RV
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
t0 = time.time()
X = np.stack([load(p) for p in folds.patient])                 # folds.csv order
print(f'loaded {X.shape} in {time.time() - t0:.0f}s')

cfg = OmegaConf.load(os.path.join(os.path.dirname(cinema.__file__), 'classification', 'acdc', 'config.yaml'))
cfg.model.views = ['sax']
cfg.model.ckpt_path = None
assert list(cfg.data.pathology) == classes, (list(cfg.data.pathology), classes)

def inner_split(tr_idx):                                       # identical to step1 (same patients)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    a, b = next(sss.split(tr_idx, y[tr_idx]))
    return tr_idx[a], tr_idx[b]

P = np.zeros((len(y), len(classes)))
curves, picks = [], []

# ------------------------------------------------ frozen probe
if MODE == 'frozen':
    model = ConvViT.from_pretrained(config=cfg, freeze=True).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(len(X)):
            xd = model.feature_forward(image_dict={'sax': torch.from_numpy(X[i:i + 1]).to(dev)}, mask_dict=None)
            cls_tok = xd['cls'].reshape(1, -1)
            tok = xd['sax'].reshape(1, -1, xd['sax'].shape[-1]).mean(1)
            feats.append(torch.cat([cls_tok, tok], 1).cpu().numpy()[0])
    Fm = np.stack(feats)
    pd.DataFrame(Fm).assign(patient=folds.patient.values).set_index('patient').to_csv(f'{DRIVE}/cinema_features.csv')  # used by step 5
    for k in range(5):
        tr, te = f != k, f == k
        grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
                            {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]},
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
        grid.fit(Fm[tr], y[tr])
        P[te] = grid.predict_proba(Fm[te])
        picks.append(dict(fold=k, C=grid.best_params_['logisticregression__C']))
        print('fold', k, grid.best_params_)

# ------------------------------------------------ full fine-tuning
if MODE == 'finetune':
    from cinema.convvit import param_groups_lr_decay
    from cinema.optim import GradScaler, adjust_learning_rate
    # T4 GPUs have no native bfloat16: use float16 unless AMP=bf16 is set explicitly
    amp_dtype = torch.bfloat16 if os.environ.get('AMP', 'fp16') == 'bf16' else torch.float16
    EPOCHS, BATCH = int(os.environ.get('EPOCHS', 100)), int(os.environ.get('BATCH', 4))
    LR = float(os.environ.get('LR', cfg.train.lr * BATCH / 64))   # their lr is for batch 64
    WARM, EVAL, PATIENCE = 10, 5, int(os.environ.get('PATIENCE', 5))
    print(f'EPOCHS {EPOCHS} | BATCH {BATCH} | LR {LR:.2e} | PATIENCE {PATIENCE} | ROI {ROI} | amp {amp_dtype}')

    def augment(x):                                             # their transforms, done on the GPU
        B = x.size(0)
        if torch.rand(1) < cfg.transform.prob:                  # contrast (gamma)
            g = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(*cfg.transform.gamma)
            x = x.clamp(min=0) ** g
        if torch.rand(1) < cfg.transform.prob:                  # Gaussian noise
            x = x + 0.1 * torch.randn_like(x)
        if torch.rand(1) < cfg.transform.prob:                  # in-plane rotation, translation, scaling
            th = torch.empty(B, device=x.device).uniform_(-np.pi, np.pi)
            s = 1 + torch.empty(B, device=x.device).uniform_(-cfg.transform.scale_range, cfg.transform.scale_range)
            t = torch.empty(B, 2, device=x.device).uniform_(-60, 60) / (PATCH[0] / 2)
            A = torch.stack([torch.stack([torch.cos(th) * s, -torch.sin(th) * s, t[:, 0]], 1),
                             torch.stack([torch.sin(th) * s, torch.cos(th) * s, t[:, 1]], 1)], 1)
            C, H, W, D = x.shape[1:]
            z = x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W)
            z = F.grid_sample(z, F.affine_grid(A, z.shape, align_corners=False), align_corners=False)
            x = z.reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)
        return x

    @torch.no_grad()
    def predict(model, idx):
        model.eval()
        out = []
        for i in idx:
            with torch.autocast('cuda', dtype=amp_dtype, enabled=dev == 'cuda'):
                out.append(model({'sax': torch.from_numpy(X[i:i + 1]).to(dev)}).float().softmax(1).cpu().numpy()[0])
        return np.stack(out)

    done = set()                                                # resume after an interrupted session
    if os.path.exists(f'{OUT}/model_selection.csv') and os.path.exists(f'{OUT}/oof_probs_partial.npy'):
        picks = pd.read_csv(f'{OUT}/model_selection.csv').to_dict('records')
        done = {int(r['fold']) for r in picks}
        P = np.load(f'{OUT}/oof_probs_partial.npy')
        if os.path.exists(f'{OUT}/learning_curves.csv'):
            curves = pd.read_csv(f'{OUT}/learning_curves.csv').to_dict('records')
        print('resuming, folds already done:', sorted(done))
    for k in range(5):
        if k in done:
            continue
        t0 = time.time()
        tr, va = inner_split(np.where(f != k)[0])
        te = np.where(f == k)[0]
        torch.manual_seed(SEED + k)
        model = ConvViT.from_pretrained(config=cfg, freeze=False)
        model.set_grad_ckpt(cfg.grad_ckpt)
        model.to(dev)
        opt = torch.optim.AdamW(param_groups_lr_decay(model, no_weight_decay_list=[], weight_decay=cfg.train.weight_decay,
                                                      layer_decay=cfg.train.layer_decay), lr=LR, betas=tuple(cfg.train.betas))
        scaler = GradScaler()
        best, best_ep, bad, best_state = -1.0, 0, 0, None
        n_steps = max(1, len(tr) // BATCH)
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.default_rng(SEED * 100 + k * 1000 + ep).permutation(tr)
            tl = 0.0
            for s in range(n_steps):
                adjust_learning_rate(opt, step=s / n_steps + ep, warmup_steps=WARM, max_n_steps=EPOCHS,
                                     lr=LR, min_lr=cfg.train.min_lr)
                b = perm[s * BATCH:(s + 1) * BATCH]
                xb = augment(torch.from_numpy(X[b]).to(dev))
                yb = torch.from_numpy(y[b]).to(dev)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=dev == 'cuda'):
                    loss = F.cross_entropy(model({'sax': xb}).float(), yb, label_smoothing=cfg.train.label_smoothing)
                scaler(loss, opt, clip_grad=cfg.train.clip_grad, parameters=model.parameters(), update_grad=True)
                opt.zero_grad()
                tl += loss.item()
            row = dict(fold=k, epoch=ep + 1, train_loss=tl / n_steps)
            if (ep + 1) % EVAL == 0:
                pv = predict(model, va)
                auc = roc_auc_score(y[va], pv, multi_class='ovo', average='macro', labels=range(len(classes)))
                row.update(val_auc=auc, val_bal_acc=balanced_accuracy_score(y[va], pv.argmax(1)))
                print(f'  fold {k} epoch {ep + 1}: train loss {tl / n_steps:.3f} | inner-val AUC {auc:.3f}', flush=True)
                if auc > best + 1e-4:
                    best, best_ep, bad = auc, ep + 1, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                if bad >= PATIENCE:
                    curves.append(row)
                    break
            curves.append(row)
        model.load_state_dict(best_state)
        torch.save({kk: v.half() for kk, v in best_state.items()}, f'{OUT}/cinema_finetune_fold{k}.pt')   # for XAI (C3)
        P[te] = predict(model, te)
        picks.append(dict(fold=k, best_epoch=best_ep, inner_val_auc=best))
        print(f'fold {k}: best epoch {best_ep}, inner-val AUC {best:.3f}, stopped at {ep + 1}, {time.time() - t0:.0f}s')
        pd.DataFrame(curves).to_csv(f'{OUT}/learning_curves.csv', index=False)
        pd.DataFrame(picks).to_csv(f'{OUT}/model_selection.csv', index=False)
        np.save(f'{OUT}/oof_probs_partial.npy', P)
        del model, opt
        torch.cuda.empty_cache()

# ------------------------------------------------ results (same metrics as step2)
np.save(f'{OUT}/oof_probs.npy', P)
if MODE == 'frozen':
    np.save(f'{DRIVE}/cinema_oof_probs.npy', P)              # used by steps 2b, 3, 5, 11
pd.DataFrame(picks).to_csv(f'{OUT}/model_selection.csv', index=False)
print(f'\nCineMA {MODE}: balanced acc {balanced_accuracy_score(y, P.argmax(1)):.3f} | '
      f'macro AUC (OvR) {roc_auc_score(y, P, multi_class="ovr", average="macro"):.3f}')
print('Saved to', OUT)
