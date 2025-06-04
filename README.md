# 🎯 Gamma vs Hadron Classification with ML | MAGIC Telescope Dataset

Welcome to my first Machine Learning project!  
This repository contains a complete pipeline for classifying gamma-ray and hadronic cosmic events using the MAGIC gamma telescope dataset, including preprocessing, visualization, model training, and evaluation.

---

## 📂 Project Structure
├── magic04.data # Raw dataset
├── code_1.py # Main Python script with all steps
├── README.md # Project documentation
├── /images # Folder containing generated plots
└── requirements.txt # List of required packages (optional)


---

## 🧪 Project Objective

The goal is to **classify high-energy cosmic events** as either:

- ✅ **Gamma rays** (signal)
- ❌ **Hadron rays** (background)

based on image characteristics captured by a **Cherenkov telescope** using a **Monte Carlo simulated dataset**.

---

## 📊 Dataset Description

The dataset is derived from **simulations** using the CORSIKA tool, designed to imitate what a **Cherenkov telescope** would detect when gamma rays or hadrons strike Earth’s atmosphere.

### Dataset Overview

| Attribute        | Description                    |
|------------------|--------------------------------|
| Instances        | 19,020                         |
| Features         | 10 Numerical Features          |
| Target           | Binary (gamma = 1, hadron = 0) |
| Task             | Classification                 |
| Domain           | High-Energy Astrophysics       |
| Missing Values   | None                           |

### Feature Columns

- `fLength`: Major axis of ellipse (mm)
- `fWidth`: Minor axis of ellipse (mm)
- `fSize`: Total light content (log scale)
- `fConc`, `fConc1`: Concentration of highest pixels
- `fAsym`: Asymmetry in light distribution
- `fM3Long`, `fM3Trans`: Moments (spread) along axes
- `fAlpha`: Orientation angle of ellipse
- `fDist`: Distance from center to origin

---

## 🛠️ Tools & Libraries Used

- Python 🐍
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (KNN, Linear Regression, Metrics)
- `imblearn` (Oversampling with RandomOverSampler)

---

## 🚀 ML Models Implemented

### ✅ 1. **K-Nearest Neighbors (KNN)**
- Balanced the training data using oversampling
- Tuned number of neighbors (`n_neighbors`)
- Evaluated with classification report and confusion matrix

### 📈 2. **Linear Regression (Baseline for comparison)**
- Evaluated using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score

---

## 📊 Visualizations

### 📌 1. Feature Distributions by Class
Each feature plotted for both Gamma and Hadron events to observe separability.

### 📌 2. Confusion Matrix
Displays model accuracy across true positive, false positive, etc.

### 📌 3. ROC Curve
Visual performance of classifier across thresholds.

### 📌 4. Correlation Heatmap
Heatmap to observe inter-feature relationships and multicollinearity.

---

## 📉 Results

- KNN Accuracy: *(include your result here)*
- Precision/Recall/F1 for both classes
- Regression metrics (baseline comparison)

---

## 📌 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

Install required libraries
pip install -r requirements.txt

Run the main file
python code_1.py

📚 References
Dataset: MAGIC Gamma Telescope Data (UCI)

Simulation Tool: CORSIKA

🔗 Connect with Me
📧 mohanprasad.23BCE11026@vitbhopal.ac.in

____________________________________________________________________________________________________________________________



## 📄 Dataset Description

This project uses the **MAGIC Gamma Telescope dataset**, a well-known dataset in the physics and machine learning communities for binary classification tasks.

### 🎯 Objective

The aim is to classify cosmic particle events as either:

* **Gamma rays** (signal)
* **Hadronic cosmic rays** (background)

using machine learning models trained on simulated detector outputs.

---

### 📊 Dataset Overview

| Attribute          | Details                                   |
| ------------------ | ----------------------------------------- |
| **Dataset Type**   | Multivariate                              |
| **Task**           | Classification                            |
| **Domain**         | Physics / Atmospheric Cherenkov Telescope |
| **Instances**      | 19,020                                    |
| **Features**       | 10 numerical + 1 binary target            |
| **Missing Values** | None                                      |

---

### 🧪 Scientific Background

The data were generated using **CORSIKA**, a Monte Carlo simulation program used to model **extensive air showers** initiated by cosmic rays.

Cherenkov telescopes detect high-energy gamma rays indirectly by recording **Cherenkov radiation**—light emitted when charged particles move faster than the speed of light in the atmosphere. These light pulses form **shower images** on a camera consisting of photomultiplier tubes.

* **Gamma particles** produce compact, elliptical, and central images.
* **Hadronic particles** (background noise) generate more chaotic and dispersed patterns.

These differences in pattern help in distinguishing the **signal (gamma)** from **background (hadron)**.

---

### 📐 Feature Explanation

| Feature    | Description                                                                    | Unit       |
| ---------- | ------------------------------------------------------------------------------ | ---------- |
| `fLength`  | Major axis length of the elliptical shower image                               | mm         |
| `fWidth`   | Minor axis length                                                              | mm         |
| `fSize`    | 10-log of the sum of pixel intensities (total signal size)                     | log(#phot) |
| `fConc`    | Ratio of the sum of the two brightest pixels to total signal size              | —          |
| `fConc1`   | Ratio of the brightest pixel to total signal size                              | —          |
| `fAsym`    | Asymmetry: distance of brightest pixel to image center projected on major axis | —          |
| `fM3Long`  | 3rd root of the third moment along the major axis (elongation measure)         | mm         |
| `fM3Trans` | 3rd root of the third moment along the minor axis                              | mm         |
| `fAlpha`   | Angle between the major axis and the vector to the image origin                | degrees    |
| `fDist`    | Distance from the image center to the camera center                            | mm         |

---

### 🏷️ Target Variable

* `class`: Binary classification target

  * `1` → Gamma ray event (signal)
  * `0` → Hadron event (background)

---

### 🔗 Source

Data generated using:

> D. Heck et al., *CORSIKA – A Monte Carlo code to simulate extensive air showers*, Forschungszentrum Karlsruhe, FZKA 6019 (1998).
> [CORSIKA Documentation (PDF)](http://rexa.info/paper?id=ac6e674e9af20979b23d3ed4521f1570765e8d68)



