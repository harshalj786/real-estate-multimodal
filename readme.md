## 🏠 Multimodal Real Estate Price Prediction

Predict residential property prices by fusing structured housing data with Sentinel-2 satellite imagery. The project benchmarks tabular-only, image-only and multimodal regressors, then uses Grad-CAM to surface neighborhood context that drives model confidence.

---

## 📌 Overview

Traditional real estate models thrive on tabular features (square footage, bedrooms, location), yet they overlook environmental signals such as nearby water, green density or urban sprawl. This project asks a simple question: *does high-level satellite context improve price estimation, or at least help explain predictions?*

What’s inside:

- ✅ Strong tabular baseline with feature engineering
- 🛰️ ResNet-based visual encoder for satellite tiles
- 🔗 Late-fusion multimodal regressor
- 🔍 Grad-CAM overlays for spatial interpretability

Accuracy gains remain honest—imagery mainly enhances storytelling rather than raw metrics.

---

## 📂 Dataset

### Tabular
- **Source:** King County Housing Dataset
- **Target:** Sale price (trained on the log-transformed target)
- **Core features:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, latitude/longitude, neighborhood averages (sqft_living15, sqft_lot15)
- **Engineered signals:** house_age, basement_ratio, living_lot_ratio, living_vs_neighbors, is_renovated

### Imagery
- **Source:** Sentinel-2 tiles fetched by latitude/longitude
- **Resolution:** ~10 m/pixel (neighborhood scale)
- **Captures:** tree cover, shoreline proximity, urban density—not house facades

---

## 🧠 Models & Results

| Model | Architecture | RMSE | Notes |
| :--- | :--- | :--- | :--- |
| **Tabular-only** | MLP (Engineered Features) | **~0.31** | 🏆 **Best Model**. Captures precise property details. |
| **Multimodal** | ResNet-18 + MLP | ~0.45 | Adds visual context but introduces noise. |
| **Image-only** | ResNet-18 (Visual) | ~1.32 | Satellite resolution (10m) is too coarse for pricing. |

> **Insight:** Tabular features drive accuracy. Imagery is best used for **explainability** (identifying water/density) rather than improving raw error metrics.

---

## 🔍 Explainability

Grad-CAM heatmaps are generated from the multimodal model’s convolutional backbone. Observed behavior:

- Highlights shorelines, rivers and other water bodies
- Responds to dense urban grids versus suburban sprawl
- Spreads attention across neighborhoods rather than specific homes

Artifacts are exported to `outputs/gradcam/` for inspection.

---

## 🗂️ Repository Layout

```
real-estate-multimodal/
├─ data/
│  ├─ raw/
│  ├─ images/
│  ├─ train_processed.csv
│  └─ test_processed.csv
├─ notebooks/
│  ├─ preprocessing.ipynb
│  └─ model_training.ipynb
├─ outputs/
│  ├─ predictions.csv
│  └─ gradcam/
├─ src/
│  ├─ data_fetcher.py
│  ├─ dataset.py
│  └─ model.py
├─ requirements.txt
├─ README.md
└─ report.pdf
```

---

## ⚙️ Quickstart

1. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
2. **Preprocess data** — run `notebooks/preprocessing.ipynb`
3. **Train + Grad-CAM** — run `notebooks/model_training.ipynb`
4. **Predictions** land in `outputs/predictions.csv`

> Tip: ensure Sentinel Hub credentials are configured in `.env` before fetching imagery.

---

## 🧾 Notes

- Tabular-only model is production-ready due to the lowest RMSE.
- Multimodal variant doubles as an interpretability tool.
- Results emphasise transparency over aggressive leaderboard chasing.
- Every notebook is parameterised to run from the project root for reproducibility.