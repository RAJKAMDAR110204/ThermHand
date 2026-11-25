# ThermHand: RGB-Thermal Hand Arthritis Screening Pipeline

A multimodal computer vision and thermal screening system that detects arthritis-related hand inflammation using paired RGB video and FLIR One thermal imaging.

---

## Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: CLOUD TRAINING (KAGGLE)                      │
│                Dataset: podakantisatyajith/thermal-data-collection (71 GB)       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 1. Cell 1: Environment Setup          │
                     │    !pip install -q mediapipe          │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 2. Cell 2: Multimodal Extraction     │
                     │    - Thermal-to-Celsius conversion    │
                     │    - MediaPipe 21-joint RGB tracking  │
                     │    - Affine registration (M_global)   │
                     │    - Otsu & hysteresis hand masking   │
                     │    - Extracts 21 thermal features     │
                     │    - Saves: features_cache.pkl        │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 3. Cell 3: Anti-Overfitting LOSO      │
                     │    - 17-fold Leave-One-Subject-Out    │
                     │    - Ledoit-Wolf covariance shrinkage │
                     │    - Out-of-fold (OOF) 95% threshold  │
                     │    - Exports: model_bundle_*.json     │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 4. Cell 4: Package Artifacts          │
                     │    - Zips outputs into .zip (~400 KB) │
                     └───────────────────┬───────────────────┘
                                         │
                                         │ (Download only ~400 KB, ZERO 71 GB download!)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: LOCAL DASHBOARD (PC)                         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 5. Local Setup                        │
                     │    pip install -r requirements        │
                     │    streamlit run ui.py                │
                     └───────────────────┬───────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
   ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
   │ Option A: Instant Feature Test  │       │ Option B: Video Tracing Test    │
   │ - Reads features_cache.pkl      │       │ - Uses 1 subject folder (S17)   │
   │ - Tests S1-S17 in 0.01s         │       │ - Live thermal video playback   │
   │ - Outputs anomaly distance      │       │ - MediaPipe skeleton overlay    │
   └─────────────────────────────────┘       └─────────────────────────────────┘
```

---

## Step-by-Step Instructions

### Step 1: Run Training & Feature Extraction on Kaggle (No 71 GB Download)

Do **not** download the 71 GB dataset to your computer. Run extraction directly in Kaggle cloud where the dataset is natively stored.

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. On the right sidebar under **Input**, click **Add Input** and search for:
   ```
   podakantisatyajith/thermal-data-collection
   ```
3. In the notebook top menu, go to **Settings** $\to$ set **Internet** to **Internet On**.
4. In the top-left menu, click **File** $\to$ **Upload Notebook**, and select `vesion2-2 (2).ipynb`.
5. Run the cells in order:
   - **Cell 1**: Installs MediaPipe.
   - **Cell 2**: Auto-detects the dataset, aligns RGB-Thermal frames, and saves `features_cache.pkl`.
   - **Cell 3**: Runs the 17-Fold Leave-One-Subject-Out (LOSO) cross-validation and exports:
     - `model_bundle_loso_calibrated.json`
     - `model_bundle_excl_S17.json`
     - `loso_oof_summary.csv`
   - **Cell 4**: Zips the output files into `thermhand_outputs.zip`.

---

### Step 2: Download Output Artifacts (~400 KB Total)

1. In the Kaggle right panel under **Output** (`/kaggle/working`), locate `thermhand_outputs.zip`.
2. Click the three dots (`...`) next to the file $\to$ **Download**.
3. Extract `thermhand_outputs.zip` into your local project root:
   ```
   D:\PROJECT OF HOPE\Arthritis\
   ```
   You should now have:
   - `model_bundle_excl_S17.json`
   - `model_bundle_loso_calibrated.json`
   - `features_cache.pkl`
   - `loso_oof_summary.csv`
   - `registration.json`

---

### Step 3: Install Local Dependencies

Open your local terminal (PowerShell or Command Prompt) and run:

```powershell
cd "D:\PROJECT OF HOPE\Arthritis"
pip install streamlit opencv-python mediapipe numpy pandas scipy
```

---

### Step 4: Launch the Screening Dashboard

Start the Streamlit application:

```powershell
streamlit run ui.py
```

Your browser will automatically open to `http://localhost:8501`.

---

### Step 5: Screen Hands in the UI

You have two testing modes inside the web interface:

#### Mode 1: Instant Screening from Features Cache (Recommended — No Video Needed)
1. In the dashboard, select:
   ```
   ⚡ Screen Subject from Pre-Extracted Features (features_cache.pkl) — Instant
   ```
2. In **Select Subject to Screen**, choose any subject (e.g. `S17`, `S1`, etc.).
3. Click **Run Analysis on Selected Subject**.
4. **Output displayed immediately**:
   - Cohort distance score ($d$) against the 95th percentile cutoff ($4.56$).
   - Status badge: `WITHIN_COHORT_RANGE` (Normal) or `OUTLIER_VS_COHORT` (Anomalous).
   - Top contributing joint temperature deviations and bilateral asymmetries.

#### Mode 2: Live Video & Skeleton Overlay (Optional)
If you want to view live playback and hand skeleton tracking:
1. Download only **one single subject's folder** (e.g., `S17`, ~150 MB) from Kaggle using this 2-line snippet in Kaggle:
   ```python
   import shutil
   shutil.make_archive('/kaggle/working/S17', 'zip', '/kaggle/input/datasets/podakantisatyajith/thermal-data-collection/FLIROne/S17')
   ```
2. Download `S17.zip` and extract it to:
   ```
   D:\PROJECT OF HOPE\Arthritis\S17\
   ```
3. In the UI, switch to:
   ```
   📹 Process Raw Video & Thermal Folder (.npy / .mp4)
   ```
4. Enter `S17` in the folder field and click **Run Analysis** to see synchronized RGB video and thermal false-color tracking side-by-side.

---

## Project File Layout

```
D:\PROJECT OF HOPE\Arthritis\
├── vesion2-2 (2).ipynb            # Kaggle extraction, LOSO training & packaging notebook
├── ui.py                           # Interactive Streamlit screening dashboard
├── model_bundle_excl_S17.json      # Calibrated production model bundle (OOF cut = 4.56)
├── model_bundle_loso_calibrated.json # Calibrated 17-fold LOSO model bundle
├── features_cache.pkl              # 21 extracted hand thermal features across 68 recordings
├── loso_oof_summary.csv            # Full out-of-fold prediction results (64 normal, 4 outliers)
├── registration.json               # Multimodal RGB-to-thermal affine matrix (M_global)
├── hand_landmarker.task            # MediaPipe 21-joint landmark neural network weights
├── pyrightconfig.json              # IDE linter configuration
└── README.md                       # Operational steps and pipeline documentation
```

---

## Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | Run `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'mediapipe'` | MediaPipe not installed | Run `pip install mediapipe` |
| `Subject folder not found: S17` | Raw video folder not on local PC | Switch UI mode to `⚡ Screen Subject from Pre-Extracted Features` |
| `Task file not found: hand_landmarker.task` | MediaPipe model asset missing | Download [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) into project root |
| Editor red squiggly on `import cv2` | Editor language server cache | Press `Ctrl+Shift+P` $\to$ select `Python: Restart Language Server` |
