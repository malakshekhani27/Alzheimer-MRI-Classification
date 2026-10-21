# 🧠 Alzheimer's Disease Classification Using Deep Learning

An AI-based MRI image classification web application built using **TensorFlow, EfficientNet-B0, and Streamlit**. The project classifies brain MRI images into four categories and provides prediction probabilities, Grad-CAM visualizations, and downloadable reports.

## 🚀 Features

- Upload a brain MRI image through a web interface.
- Predict one of four dataset categories:
  - Mild Demented
  - Moderate Demented
  - Non Demented
  - Very Mild Demented
- Display prediction confidence and class probabilities.
- Generate a Grad-CAM heatmap to visualize image regions influencing the model's prediction.
- Download prediction reports and Grad-CAM images.
- Download a complete PDF analysis report.
- View a confusion matrix generated during model evaluation.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- EfficientNet-B0
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib
- ReportLab

## 📁 Project Structure

```text
Alzheimer-MRI-Classification/
├── app.py
├── config.py
├── model.py
├── data_loader.py
├── train.py
├── evaluate.py
├── predict.py
├── gradcam.py
├── prepare_dataset.py
├── convert_dataset.py
├── requirements.txt
├── models/
│   ├── alzheimer_efficientnet.keras
│   └── class_names.txt
├── outputs/
│   └── confusion_matrix.png
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/divyadewangan0804/Alzheimer-MRI-Classification.git
cd Alzheimer-MRI-Classification
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install reportlab
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 🧪 Model Evaluation

The project uses a trained EfficientNet-B0-based image classification model.

The current recorded test accuracy is approximately **58.05%**. Performance varies by class, and the model requires further improvement before any practical clinical use.

Evaluation outputs include a classification report and confusion matrix.

## 🔍 Explainable AI with Grad-CAM

Grad-CAM generates a heatmap highlighting image regions that influenced the model's prediction. This provides a visual explanation of model behavior; it is not a verified medical lesion map.

## 📦 Dataset

The dataset contains four MRI image categories. The dataset directory is excluded from this Git repository, so users must obtain the dataset separately and arrange it according to the project's expected folder structure before training or evaluation.

Please verify the dataset's license and sharing conditions before redistributing any images.

## ⚠️ Limitations and Medical Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic device and must not be used to diagnose Alzheimer's disease or guide treatment. Predictions may be incorrect. Consult qualified healthcare professionals for medical assessment.

## 👩‍💻 Author

**Divya Dewangan**

GitHub: [@divyadewangan0804](https://github.com/divyadewangan0804)

