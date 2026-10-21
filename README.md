# Alzheimer's Disease Classification from MRI Images Using Deep Learning

A complete educational/research starter project that classifies MRI image files into:
- CN: Cognitively Normal
- MCI: Mild Cognitive Impairment
- AD: Alzheimer's Disease

It uses EfficientNetB0 transfer learning, patient-independent folder-based image datasets,
evaluation metrics, confusion matrix, prediction, and Grad-CAM visualization through Streamlit.

## Important
This is a research/educational prototype, NOT a medical diagnostic device.

## 1. Requirements
- Windows 10/11
- Python 3.11
- VS Code recommended
- A GPU is optional. CPU training is possible but slower.

## 2. Setup

PowerShell:
```powershell
cd Alzheimer-MRI-Classification
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Dataset format

Put your licensed/publicly obtained image dataset under `data/raw/`:

```text
data/
└── raw/
    ├── CN/
    │   ├── image001.jpg
    │   └── ...
    ├── MCI/
    │   ├── image101.jpg
    │   └── ...
    └── AD/
        ├── image201.jpg
        └── ...
```

Accepted image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

Do NOT claim a dataset is ADNI/OASIS unless it actually came from that source and you have the appropriate access/permission.

## 4. Split dataset

```powershell
python prepare_dataset.py --input data/raw --output data/split --train 0.70 --val 0.15 --test 0.15
```

This creates:

```text
data/split/
├── train/
├── val/
└── test/
```

The script uses a deterministic random seed.

## 5. Train

```powershell
python train.py
```

The trained model is saved to:
`models/alzheimer_efficientnet.keras`

Training history is saved to:
`outputs/training_history.csv`

Plots are saved to:
`outputs/training_curves.png`

## 6. Evaluate

```powershell
python evaluate.py
```

Outputs:
- accuracy
- classification report
- confusion matrix
- confusion matrix image

## 7. Predict one image

```powershell
python predict.py --image "path\to\mri.jpg"
```

## 8. Run web app

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 9. Project pipeline

MRI image
-> preprocessing
-> EfficientNetB0
-> 3-class prediction
-> confidence
-> Grad-CAM
-> Streamlit interface

## 10. Notes on medical imaging
The supplied implementation is intentionally image-file based so it can run on common public
2D MRI image datasets. If your source data is NIfTI/DICOM, convert/process it according to the
dataset's official documentation before placing the resulting images in the class folders.

For a research-grade system, split data at the patient level rather than allowing slices from
the same patient to appear in both training and testing. External validation on an independent
dataset is also recommended.
