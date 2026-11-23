# ThermHand Screening (Version 2.0)
### Multimodal RGB-Thermal Screening & Cohort Anomaly Detection for Hand Arthritis

ThermHand is a non-invasive screening pipeline designed to detect localized inflammation and physiological anomalies associated with arthritis using paired RGB and radiometric thermal imaging (FLIR One). 

The system leverages MediaPipe hand landmark tracking, multimodal spatial registration, morphological hand segmentation, anatomical thermal feature extraction (including bilateral asymmetry), and an unsupervised **Subspace + Residual (SPE) Anomaly Scorer** trained on a 17-subject cohort.

---

## 1. System Architecture

```
[ Paired Video Capture ]
   ├── Thermal Stream: 120x160 radiometric matrices (.npy)
   └── RGB Stream: 640x480 video (.mp4)
              │
              ▼
[ Multimodal Preprocessing ]
   ├── Kelvin / Raw Counts -> Celsius Calibration
   ├── Thermal Camera Orientation Normalization
   └── 2D Hand Landmark Tracking (MediaPipe Tasks HandLandmarker)
              │
              ▼
[ Affine Registration & Hand Masking ]
   ├── Affine Transformation (M_global) mapping RGB joints -> Thermal
   └── Adaptive Otsu + Hysteresis Morphology Hand Segmentation
              │
              ▼
[ Automated Quality Gate ]
   ├── Frame count (n >= 3)
   ├── Mask stability (q_mask_ok_frac >= 0.85)
   ├── Area variation (q_area_cv <= 0.35)
   └── Valid joint detection rate (q_valid_joint_frac >= 0.50)
              │
              ▼
[ Thermal Feature Extraction (21 Features) ]
   ├── Global & Joint-Level Heat Metrics
   ├── Physiological Finger-to-Palm Temperature Gradients
   └── Bilateral Left-Right Absolute Asymmetry (_lr_abs)
              │
              ▼
[ Cohort Anomaly Scorer (SubspaceResidualScorer) ]
   ├── Robust Median Imputation & MAD Standardization
   ├── PCA Dimensionality Reduction (Subspace)
   ├── In-Subspace Minimum Covariance Determinant (MinCovDet Mahalanobis)
   └── Out-of-Subspace Squared Prediction Error (SPE Residual)
              │
              ▼
[ Model Export & Interactive Dashboard (ui.py) ]
   ├── Serialized Model Bundle: model_bundle_excl_S17.json
   └── Streamlit Clinical Inspection UI
```

---

## 2. Dataset & Cohort Structure

The pipeline is built on a 17-subject cohort dataset collected using FLIR One thermal cameras:
- **Kaggle Dataset**: `podakantisatyajith/thermal-data-collection` (FLIROne directory)
- **Cohort**: 17 unique human subjects (`S1` through `S17`)
- **Streams**: Left and right hand recordings across experimental tasks
- **Holdout Setup (Version 2)**:
  - **Training Subjects**: 16 subjects (`S1` to `S16`, ~62 quality-gated recordings)
  - **Held-Out Test Subject**: 1 subject (`S17`) held out for out-of-cohort testing
  - **Threshold**: Outlier cutoff set at the cohort 95th percentile ($d_{\text{cut\_p95}}$)

---

## 3. Key Pipeline Modules

### 3.1. Multimodal Registration & MediaPipe Tracking
Thermal cameras and RGB lenses have different fields of view and optical axes. The pipeline computes a global affine matrix $M_{\text{global}}$:
$$\begin{bmatrix} x_{\text{thermal}} \\ y_{\text{thermal}} \end{bmatrix} = M_{\text{global}} \begin{bmatrix} x_{\text{rgb}} \\ y_{\text{rgb}} \\ 1 \end{bmatrix}$$
MediaPipe detects 21 hand landmarks on RGB frames (wrist, MCP, PIP, DIP, and finger tips). These 2D landmark coordinates are warped onto the $120 \times 160$ thermal space to localize anatomical joint regions (MCP and PIP joints).

### 3.2. Hand Segmentation & Quality Gating
- Hand masks are extracted using Otsu adaptive thresholding followed by morphological hole filling and hysteresis filtering.
- Recordings must pass a pre-registered **Quality Gate** before scoring:
  - `n_used >= 3`: At least 3 valid analysis frames.
  - `q_mask_ok_frac >= 0.85`: Consistent hand silhouette without excessive dropouts.
  - `q_area_cv <= 0.35`: Stable hand position without violent movement.
  - `q_valid_joint_frac >= 0.50`: MediaPipe successfully tracked joints in $\ge 50\%$ of frames.

### 3.3. Anatomical Feature Vector (21 Features)
The model extracts 21 engineered thermal indicators per hand:

| Feature Category | Feature Column | Description |
| :--- | :--- | :--- |
| **Mask Statistics** | `f_mask_mean_c` | Mean hand surface temperature (°C) |
| | `f_mask_p90_c` | 90th percentile temperature (°C) |
| | `f_mask_sd_c` | Temperature standard deviation |
| | `f_mask_grad_norm_c` | Surface spatial temperature gradient magnitude |
| | `f_mask_p99_minus_med_c` | Upper tail thermal spread (p99 - median) |
| **Joint & Focal Heat** | `f_focal_peak_c` | Peak localized hyperthermia |
| | `f_local_joint_contrast_c` | Temperature contrast of joints vs surrounding tissue |
| | `f_joint_p90_contrast_c` | 90th percentile joint heat contrast |
| | `f_peak_joint_excess_c` | Maximum excess heat localized over MCP joints |
| | `f_drift_slope_c` | Temporal thermal drift slope across video frames |
| | `f_step_noise_c` | Inter-frame sensor thermal noise |
| **Anatomical Gradients** | `f_palm_minus_tip_c` | Temperature difference: palm center minus finger tips |
| | `f_index_mcp_minus_tip_c` | Index finger MCP joint minus index tip |
| | `f_middle_mcp_minus_tip_c`| Middle finger MCP joint minus middle tip |
| | `f_ring_mcp_minus_tip_c` | Ring finger MCP joint minus ring tip |
| | `f_pinky_mcp_minus_tip_c` | Pinky finger MCP joint minus pinky tip |
| **Bilateral Asymmetry** | `f_mask_mean_c_lr_abs` | Absolute Left vs Right mean temperature difference |
| | `f_focal_peak_c_lr_abs` | Absolute Left vs Right focal peak difference |
| | `f_local_joint_contrast_c_lr_abs` | Absolute Left vs Right joint contrast difference |
| | `f_joint_p90_contrast_c_lr_abs` | Absolute Left vs Right joint p90 contrast difference |
| | `f_palm_minus_tip_c_lr_abs` | Absolute Left vs Right palm-to-tip gradient difference |

---

## 4. Cohort Anomaly Detection Model (`SubspaceResidualScorer`)

Because clinical ground-truth diagnostic labels are scarce, ThermHand employs an unsupervised cohort-deviation architecture:

1. **Robust Scaling**:
   - Impute missing features with cohort median $\tilde{X}$.
   - Center by median $\mu_{1/2}$ and scale by Median Absolute Deviation (MAD):
     $$Z = \text{clip}\left(\frac{X - \mu_{1/2}}{\max(1.4826 \cdot \text{MAD}(X), 0.5 \sigma, 10^{-3})}, -6.0, 6.0\right)$$
2. **Subspace Projection (PCA)**:
   - Projects normalized features $Z$ into a lower-dimensional subspace capturing $\ge 90\%$ of cohort variance ($2 \le k \le 6$ components):
     $$P = (Z - \bar{Z}) V_k$$
3. **In-Subspace Mahalanobis Distance**:
   - Fits Minimum Covariance Determinant (`MinCovDet`, support fraction 0.8) on $P$:
     $$d_{\text{sub}} = \sqrt{(P - \hat{\mu})^T \hat{\Sigma}^{-1} (P - \hat{\mu})}$$
4. **Out-of-Subspace Squared Prediction Error (SPE)**:
   - Measures thermal patterns orthogonal to normal cohort variance:
     $$d_{\text{res}} = \sqrt{\frac{1}{D} \|Z - \hat{Z}\|^2}, \quad \hat{Z} = \bar{Z} + P V_k^T$$
5. **Combined Distance Metric**:
   $$z_{\text{sub}} = \max\left(0, \frac{d_{\text{sub}} - \text{med}_{\text{sub}}}{\text{IQR}_{\text{sub}}}\right), \quad z_{\text{res}} = \max\left(0, \frac{d_{\text{res}} - \text{med}_{\text{res}}}{\text{IQR}_{\text{res}}}\right)$$
   $$d_{\text{comb}} = \sqrt{z_{\text{sub}}^2 + w \cdot z_{\text{res}}^2} \quad (w = 0.5)$$

If $d_{\text{comb}} > d_{\text{cut\_p95}}$, the recording is flagged as `OUTLIER_VS_COHORT`; otherwise, it is classified as `WITHIN_COHORT_RANGE`.

---

## 5. Repository File Structure

```
.
├── vesion2-2 (2).ipynb               # Complete Jupyter notebook (Kaggle pipeline)
│   ├── Cell 1: Environment setup & MediaPipe installation
│   ├── Cell 2: Feature extraction, registration, gate checks & cohort fitting
│   └── Cell 3: Held-out model bundle exporter (S17 holdout)
├── model_bundle_excl_S17.json        # Exported JSON bundle (trained on S1-S16, S17 excluded)
├── ui.py                             # Interactive Streamlit screening dashboard
├── RGB_Thermal_Arthritis_Screening_v18.pptx # Project overview & technical presentation
└── README.md                         # Technical documentation
```

---

## 6. How to Run the Pipeline

### Option A: Running the Training Notebook (`vesion2-2 (2).ipynb`) on Kaggle
1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and click **Upload Notebook**.
2. Upload `vesion2-2 (2).ipynb`.
3. In the right panel, attach the dataset: `podakantisatyajith/thermal-data-collection`.
4. Enable **Internet On** in the notebook settings (required to fetch MediaPipe landmarker weights).
5. Run the cells in order:
   - **Cell 1**: Installs `mediapipe`.
   - **Cell 2**: Executes registration, processes video frames, extracts features, and creates `features_cache.pkl`.
   - **Cell 3**: Fits the cohort model excluding `S17` and exports `model_bundle_excl_S17.json`.
6. Download `model_bundle_excl_S17.json` from the `/kaggle/working` directory.

### Option B: Running the Streamlit Dashboard (`ui.py`) Locally
The dashboard provides a clinical inspection interface for reviewing hand thermal recordings.

1. **Install Requirements**:
   ```bash
   pip install streamlit opencv-python numpy pandas scipy mediapipe
   ```
2. **Launch Streamlit**:
   ```bash
   streamlit run ui.py
   ```
3. **Using the Interface**:
   - **Model Bundle JSON Path**: Defaults to `model_bundle_excl_S17.json`.
   - **Held-Out Subject Folder**: Path to the test subject's recordings (e.g. `S17`).
   - The UI displays:
     - Side-by-side RGB and Thermal playback with joint overlays.
     - Frame-by-frame quality checks and temperature profiles.
     - Cohort z-score deviation bars for each anatomical joint.
     - Final screening status (`WITHIN_COHORT_RANGE`, `OUTLIER_VS_COHORT`, or `RECAPTURE_RECOMMENDED`).

---

## 7. Model Bundle Specification (`model_bundle_excl_S17.json`)

The exported JSON bundle contains all state necessary for standalone inference without re-running training:

| Field | Description |
| :--- | :--- |
| `schema_version` | `"6.1.0-single-subject-holdout"` |
| `heldout_subject` | `"S17"` |
| `train_subjects` | List of 16 training subject IDs (`["S1", ..., "S16"]`) |
| `n_train_recordings` | Number of gate-passed training recordings (62) |
| `feature_columns` | List of 21 extracted feature column names |
| `impute` | 21-element imputation vector (medians) |
| `center` / `scale` | 21-element median and MAD scaling vectors |
| `pca_mean` / `pca_components` | PCA projection matrix and mean vector |
| `mcd_location` / `mcd_precision` | MinCovDet center and precision ($\Sigma^{-1}$) matrices |
| `sub_med` / `sub_iqr` | Subspace distance median and IQR normalization constants |
| `res_med` / `res_iqr` | Residual distance median and IQR normalization constants |
| `d_cut_p95` | 95th percentile anomaly decision boundary (e.g. `5.3672`) |
| `quality_gate` | Quality filter thresholds (`min_n_used`, `min_mask_ok_frac`, etc.) |
| `M_global` | $2 \times 3$ affine RGB-to-thermal registration matrix |
| `thermal_orient` | Camera orientation flag (`"rot_ccw"`) |
| `disclaimer` | `"relative to the 17-subject cohort, not a diagnosis"` |

---

## 8. Clinical Disclaimer
> **Notice**: ThermHand is an investigative cohort anomaly screening tool designed for research and exploratory screening relative to a 17-subject baseline cohort. It is not an FDA-approved medical diagnostic device and does not provide a definitive medical diagnosis.
