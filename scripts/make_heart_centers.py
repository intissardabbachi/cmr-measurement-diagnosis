# Heart centre per patient from the OUT-OF-FOLD predicted masks of step 3
# (each patient's mask comes from the U-Net of the fold where it was a test patient,
#  so no ground truth and no leakage). Output: heart_centers.csv (native voxel indices, ED).
import glob, os, numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
rows = []
for f in sorted(glob.glob(f'{DRIVE}/step3_challenge_baseline/pred_masks/*.npz')):
    m = np.load(f)['ED'] > 0                     # LV + MYO + RV
    cx, cy, cz = np.argwhere(m).mean(0)
    rows.append(dict(patient=os.path.basename(f)[:-4], cx=cx, cy=cy, cz=cz))
df = pd.DataFrame(rows)
assert len(df) == 100, len(df)
df.to_csv(f'{DRIVE}/heart_centers.csv', index=False)
print(df.head(), '\nsaved', len(df), 'centres to', f'{DRIVE}/heart_centers.csv')
