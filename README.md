
# 🌍 Land Use Classification with EuroSAT and MobileNetV2

This project performs land use classification using satellite imagery from the [EuroSAT RGB dataset](https://github.com/phelber/eurosat). It leverages **Convolutional Neural Networks (CNNs) and Deep Learning techniques** through **Transfer Learning** with the **MobileNetV2** architecture. The model is fine-tuned to recognize 10 distinct land cover classes, demonstrating the effectiveness of pre-trained networks in geospatial image classification tasks.

## 📁 Project Structure

```
project/
├── models/
│   └── eurosat_mobilenetv2.h5     # Trained classification model (optional to upload)
├── notebook/
│   └── final_model.ipynb          # Full training pipeline (EDA, modeling, evaluation)
├── README.md
└── requirements.txt               # Required Python packages
```

## 🧠 Model Overview

- **Base architecture**: MobileNetV2 (pre-trained on ImageNet)
- **Input size**: 128x128 RGB images
- **Training strategy**: 
  - Early layers frozen
  - Fine-tuning last 30 layers
  - Data augmentation applied
- **Performance**: 93.94% validation accuracy with high precision and recall across 10 classes

## 🗂️ Dataset

- **Source**: EuroSAT RGB (10 land use/land cover classes)
- **Split**: 80% training / 20% validation (via `ImageDataGenerator`)

## 🧪 Evaluation

- Confusion matrix and classification report included
- Final accuracy: **93.94%**
- Weighted F1-score: **~0.94**

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `notebook/final_model.ipynb`
4. Optional: load model from `models/eurosat_mobilenetv2.h5` if not retraining

## 📦 Dependencies

- TensorFlow / Keras
- NumPy / Matplotlib / Seaborn
- Scikit-learn

## 📌 Notes

- The model file (`.h5`) is optional in the repository due to GitHub size constraints.
- You can retrain the model from scratch using the provided notebook.

## 📸 Example Predictions

Visual results and confusion matrix are shown in the notebook output.

---

## 🚀 Possible Future Improvements

- Use class balancing or focal loss to handle confusion in minority classes.
- Incorporate multispectral bands (e.g., NDVI) for deeper vegetation separation.
- Deploy the model with an interactive prediction dashboard (e.g., Streamlit).

**Author:** [Pedro Rios] 
**LinkedIn:** [https://www.linkedin.com/in/pedro-rios-a7a20733b/]
