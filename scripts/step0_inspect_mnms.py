# =============================================================================
# STEP 0 — Inspect the M&Ms-1 zip WITHOUT extracting it (read-only)
# Purpose: learn the folder layout, the metadata file, vendors and pathology
# labels, so the external-test protocol can be written and locked.
# No model is run here; nothing is evaluated on M&Ms-1.
# CPU is enough. Needs a shortcut to M&M_dataset.zip inside My Drive.
# =============================================================================
import os, glob, zipfile, io, collections
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

DRIVE = '/content/drive/MyDrive/CMR_concept_bottleneck'
cands = glob.glob('/content/drive/MyDrive/**/M&M_dataset.zip', recursive=True)
print('zip found at:', cands)
assert cands, 'Add a shortcut to M&M_dataset.zip in My Drive (CMR_concept_bottleneck) first'
ZIP = cands[0]

z = zipfile.ZipFile(ZIP)
names = z.namelist()
print('entries in zip:', len(names))

# folder layout, 3 levels deep
tree = collections.Counter('/'.join(n.split('/')[:3]) for n in names)
print('\nTop of the layout (path : number of entries):')
for k, v in sorted(tree.items())[:60]:
    print(f'  {k} : {v}')

# file types
ext = collections.Counter(os.path.splitext(n)[1].lower() if not n.endswith('.gz') else '.nii.gz' for n in names)
print('\nFile types:', dict(ext))

# subject folders per split
subj = collections.defaultdict(set)
for n in names:
    parts = n.strip('/').split('/')
    if n.endswith('.nii.gz') and len(parts) >= 3:
        subj['/'.join(parts[:-2])].add(parts[-2])
print('\nSubjects per folder:')
for k, v in sorted(subj.items()):
    print(f'  {k}: {len(v)} subjects, e.g. {sorted(v)[:2]}')

# example file names in one subject
ex = [n for n in names if n.endswith('.nii.gz')][:6]
print('\nExample image files:', ex)

# metadata tables (csv / xlsx), read straight from the zip
meta = [n for n in names if n.lower().endswith(('.csv', '.xlsx', '.xls'))]
print('\nMetadata files:', meta)
for m in meta:
    raw = z.read(m)
    df = pd.read_csv(io.BytesIO(raw)) if m.lower().endswith('.csv') else pd.read_excel(io.BytesIO(raw))
    print(f'\n=== {m}: {df.shape}')
    print('columns:', list(df.columns))
    print(df.head(3).to_string())
    for col in df.columns:
        if any(w in col.lower() for w in ('vendor', 'centre', 'center', 'patholog', 'diagnos', 'group')):
            print(f'\n  value counts of {col}:')
            print(df[col].value_counts(dropna=False).to_string())
