# 🧬 Histopathology Image Classification with Machine Learning

This project explores the classification of histopathological images from the **PathMNIST** dataset, focusing on evaluating preprocessing techniques and comparing the performance of **Gradient Boosting**, **Multilayer Perceptron (MLP)**, and **Convolutional Neural Networks (CNN)**.

---

## 📊 Dataset Overview

* **Source**: [MedMNIST2D – PathMNIST](https://medmnist.com/)
* **Samples**:

  * `X_train.npy`: 32,000 images
  * `X_test.npy`: 8,000 images
* **Image Format**: 28×28 RGB patches from HE-stained CRC slides
* **Classes**: 9 tissue types (labeled 0–8)

---

## 🔍 Key Insights from Data Exploration

* **Pixel Distribution**:

  * Left-skewed brightness, dominated by red and blue channels.
  * Statistically significant differences across RGB channels → motivated **channel-wise normalization**.

* **Class Distribution**:

  * Minor class imbalance (max/min ratio ≈ 1.7)
  * T-SNE projection shows:

    * Class 1 is highly separable.
    * Classes 2, 5, 7 heavily overlap.

---

## 🛠️ Preprocessing Techniques

| Technique           | Description                                              | MLP Accuracy | CNN Accuracy |
| ------------------- | -------------------------------------------------------- | ------------ | ------------ |
| Channel-wise Z-Norm | Normalize each RGB channel separately                    | 60.70%       | 85.80%       |
| Color Deconvolution | Separate overlapping HE stains (histopathology-specific) | 55.06%       | 85.12%       |
| CLAHE               | Local contrast enhancement                               | 60.54%       | 81.04%       |
| HSV Conversion      | Reflect color, saturation, brightness                    | 61.51%       | 84.51%       |
| LAB Color Space     | Better color separation mimicking human vision           | **64.57%**   | 86.19%       |
| Edge Enhancement    | Add Sobel edges as 4th channel                           | 61.92%       | 84.26%       |
| **Best Combo**      | LAB for MLP, HSV+Edge for CNN                            | **64.94%**   | **87.67%**   |

---

## 🧠 Models and Performance

### 🔹 Gradient Boost (XGBoost)

* Best Test Accuracy: **58.83%**
* Strengths: Fast, interpretable for tabular features
* Weaknesses: Poor at spatial patterns in raw images

### 🔸 Multilayer Perceptron (MLP)

* Best Test Accuracy: **69.82%**
* Architecture: 3-layer dense (256-512-256), BatchNorm, Dropout
* Limitation: Loses spatial structure by flattening images

### 🔷 Convolutional Neural Network (CNN)

* Best Test Accuracy: **91.86%**
* Architecture:

  * 2–3 convolutional blocks (32 → 64 → 256 filters)
  * Dense head with dropout
* Best performance by preserving spatial hierarchies and leveraging enhancement

---

## 🎯 Final Comparison

| Model          | Test Accuracy | Notes                          |
| -------------- | ------------- | ------------------------------ |
| Gradient Boost | 58.83%        | Limited spatial learning       |
| MLP            | 69.82%        | Decent generalization          |
| CNN            | **91.86%**    | Best result, image-aware model |

---

## 🚀 Future Work

* **Advanced Augmentation**: Try **AutoAugment**, **RandAugment**, or **AugMix**.
* **Better Feature Engineering**: Use **LDA** to improve class separability.
* **Stronger Models**: Apply **pretrained networks** (e.g., ResNet, EfficientNet).
* **Scalable Tuning**: Expand Bayesian tuning with parallel GPU jobs.

---

## 📁 Project Structure

```bash
├── data/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── preprocessing_experiments.ipynb
├── models/
│   ├── cnn.py
│   ├── mlp.py
│   └── gradient_boost.py
└── README.md
```

---

## 📚 References

* Yang et al., 2023. *MedMNIST v2: A large-scale lightweight benchmark for medical image classification.*
* Kather et al., 2019. *Predicting survival from colorectal cancer histology slides.*
* Cooper et al., 2023. *Computational Pathology and Diagnostic Automation.*
* Schneider & Xhafa, 2022. *Class Imbalance in Deep Learning.*
* Gowda et al., 2019. *Color Space Techniques in Medical Imaging.*
