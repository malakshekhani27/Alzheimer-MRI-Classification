from pathlib import Path
import io
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage

from config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES_PATH
from gradcam import make_gradcam_heatmap, overlay_heatmap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
    }

    .main-header {
        background: linear-gradient(135deg, #173f5f, #20639b);
        padding: 28px 35px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 5px 18px rgba(0,0,0,0.08);
    }

    .main-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 8px;
        font-size: 16px;
        opacity: 0.9;
    }

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5eaf0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        margin-bottom: 18px;
    }

    .prediction-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #dce5ee;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }

    .prediction-label {
        font-size: 14px;
        color: #667085;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 28px;
        font-weight: 700;
        color: #173f5f;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #173f5f;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .footer {
        text-align: center;
        color: #667085;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return [
            x.strip()
            for x in CLASS_NAMES_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

    return [
        "Mild_Demented",
        "Moderate_Demented",
        "Non_Demented",
        "Very_Mild_Demented",
    ]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    return np.expand_dims(array, axis=0)


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def generate_pdf_report(image, predicted_class, confidence, class_names, probabilities, gradcam_png=None):
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Alzheimer's MRI Classification Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=19, leading=23, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"],
        fontSize=13, leading=16, spaceBefore=10, spaceAfter=8
    )
    normal_style = ParagraphStyle(
        "ReportNormal", parent=styles["BodyText"],
        fontSize=9.5, leading=13
    )

    story = []
    story.append(Paragraph("ALZHEIMER'S DISEASE MRI CLASSIFICATION REPORT", title_style))
    story.append(Paragraph(
        "Deep Learning Based MRI Image Classification with Explainable AI",
        ParagraphStyle("Subtitle", parent=normal_style, alignment=TA_CENTER,
                       fontSize=10, spaceAfter=15)
    ))

    info = [
        ["Report Date", datetime.now().strftime("%d-%m-%Y %H:%M")],
        ["Architecture", "EfficientNet-B0"],
        ["Framework", "TensorFlow / Keras"],
        ["Input Size", "224 x 224"],
        ["Number of Classes", str(len(class_names))],
    ]
    info_table = Table(info, colWidths=[150, 300])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D3DD")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([info_table, Spacer(1, 15)])

    story.append(Paragraph("1. Uploaded MRI Image", heading_style))
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    story.extend([
        PDFImage(image_buffer, width=3.5 * inch, height=3.5 * inch),
        Spacer(1, 12),
    ])

    story.append(Paragraph("2. Prediction Result", heading_style))
    result_table = Table([
        ["Predicted Classification", predicted_class.replace("_", " ")],
        ["Confidence", f"{confidence * 100:.2f}%"],
    ], colWidths=[200, 250])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D3DD")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([result_table, Spacer(1, 12)])

    story.append(Paragraph("3. Class Probabilities", heading_style))
    rows = [["Class", "Probability"]]
    for name, probability in zip(class_names, probabilities):
        rows.append([name.replace("_", " "), f"{float(probability) * 100:.2f}%"])
    probability_table = Table(rows, colWidths=[300, 150])
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173F5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D3DD")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([probability_table, Spacer(1, 12)])

    story.append(Paragraph("4. Explainable AI — Grad-CAM", heading_style))
    story.append(Paragraph(
        "Grad-CAM visualizes image regions that contributed to the selected model prediction. "
        "It should not be interpreted as a clinical lesion map.", normal_style
    ))
    story.append(Spacer(1, 8))
    if gradcam_png is not None:
        gradcam_png.seek(0)
        story.append(PDFImage(gradcam_png, width=6.5 * inch, height=2.7 * inch))
    else:
        story.append(Paragraph("Grad-CAM image was not available for this analysis.", normal_style))

    story.extend([Spacer(1, 15), Paragraph("5. Medical Disclaimer", heading_style)])
    story.append(Paragraph(
        "This application is a research and educational prototype. It is not a medical "
        "diagnostic device and must not replace professional medical evaluation. The "
        "prediction should not be used as a standalone basis for medical decisions.",
        normal_style
    ))
    story.extend([Spacer(1, 15), Paragraph(
        "Generated by Alzheimer's Disease MRI Classifier | EfficientNet-B0 + Grad-CAM",
        ParagraphStyle("FooterText", parent=normal_style, alignment=TA_CENTER,
                       fontSize=8, textColor=colors.grey)
    )])

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 Alzheimer MRI")

    st.markdown("---")

    st.markdown("### About the Project")

    st.write(
        """
        A deep-learning based research prototype for
        classifying MRI images into four Alzheimer's-related
        categories.
        """
    )

    st.markdown("### Model")

    st.info(
        """
        **Architecture:** EfficientNet-B0  
        **Framework:** TensorFlow / Keras  
        **Input Size:** 224 × 224  
        **Classes:** 4
        """
    )

    st.markdown("### Classification Classes")

    st.write("• Mild Demented")
    st.write("• Moderate Demented")
    st.write("• Non Demented")
    st.write("• Very Mild Demented")

    st.markdown("---")

    st.caption("Research / Educational Prototype")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🧠 Alzheimer's Disease MRI Classifier</h1>
        <p>
            Deep Learning Based MRI Image Classification
            with Explainable AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ This application is a research and educational prototype. "
    "It is not a medical diagnostic device and must not replace "
    "professional medical evaluation."
)


# ============================================================
# MODEL CHECK
# ============================================================

if not MODEL_PATH.exists():

    st.error("❌ Trained model not found.")

    st.info(
        "Please make sure the trained model exists in the models folder."
    )

    st.stop()


model = load_model()
class_names = load_class_names()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload MRI Image</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose an MRI image for analysis",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
    ],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIF and TIFF",
)


# ============================================================
# BEFORE UPLOAD
# ============================================================

if not uploaded:

    st.info(
        "👆 Upload an MRI image above to start the analysis."
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>How it works</h3>

        <p><b>1.</b> Upload an MRI image</p>

        <p><b>2.</b> The image is processed and resized</p>

        <p><b>3.</b> EfficientNet-B0 performs classification</p>

        <p><b>4.</b> The predicted class and confidence are displayed</p>

        <p><b>5.</b> Grad-CAM provides a visual explanation</p>

        <p><b>6.</b> Download the prediction report</p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.markdown(
        '<div class="section-title">🔍 MRI Analysis</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # LEFT: IMAGE
    # ========================================================

    with left:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True,
        )

        st.subheader("📷 Uploaded MRI")

        st.image(
            image,
            use_container_width=True,
        )

        st.caption(
            f"Image size: {image.size[0]} × {image.size[1]} pixels"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # RIGHT: ANALYSIS
    # ========================================================

    with right:

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True,
        )

        st.subheader("🔬 Model Analysis")

        st.write(
            "Click the button below to analyze the uploaded MRI image."
        )

        analyze = st.button(
            "🚀 Analyze MRI",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        if analyze:

            with st.spinner(
                "Analyzing MRI image using EfficientNet-B0..."
            ):

                array = preprocess(image)

                probabilities = model.predict(
                    array,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index]
                )


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="prediction-label">'
                'Predicted Classification'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="prediction-value">'
                f'{predicted_class.replace("_", " ")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                float(confidence)
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📊 Class Probabilities'
                '</div>',
                unsafe_allow_html=True,
            )

            for name, probability in zip(
                class_names,
                probabilities
            ):

                st.progress(
                    float(probability),
                    text=(
                        f"{name.replace('_', ' ')}: "
                        f"{probability * 100:.2f}%"
                    ),
                )


            # =================================================
            # PROBABILITY BAR CHART
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📈 Probability Chart'
                '</div>',
                unsafe_allow_html=True,
            )

            chart_data = {
                name.replace("_", " "): float(probability)
                for name, probability in zip(
                    class_names,
                    probabilities
                )
            }

            st.bar_chart(chart_data)


            # =================================================
            # DOWNLOAD PREDICTION REPORT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📥 Download Prediction Report'
                '</div>',
                unsafe_allow_html=True,
            )

            report = f"""
ALZHEIMER'S DISEASE MRI CLASSIFICATION REPORT
==============================================

Prediction Result
-----------------
Predicted Classification: {predicted_class.replace("_", " ")}
Confidence: {confidence * 100:.2f}%

Class Probabilities
-------------------
"""

            for name, probability in zip(
                class_names,
                probabilities
            ):
                report += (
                    f"{name.replace('_', ' ')}: "
                    f"{probability * 100:.2f}%\n"
                )

            report += """
----------------------------------------------
Model Information
----------------------------------------------
Architecture: EfficientNet-B0
Framework: TensorFlow / Keras
Input Size: 224 x 224
Number of Classes: 4

----------------------------------------------
Disclaimer
----------------------------------------------
This application is a research and educational
prototype. It is not a medical diagnostic device
and must not replace professional medical evaluation.
"""

            st.download_button(
                label="📥 Download Prediction Report",
                data=report,
                file_name="alzheimer_mri_prediction_report.txt",
                mime="text/plain",
                use_container_width=True,
            )


            # =================================================
            # GRAD-CAM
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🔥 Explainable AI — Grad-CAM'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Grad-CAM visualizes image regions that contributed "
                "to the selected model prediction. It should not "
                "be interpreted as a clinical lesion map."
            )


            gradcam_buffer = None

            try:

                with st.spinner(
                    "Generating Grad-CAM explanation..."
                ):

                    heatmap = make_gradcam_heatmap(
                        array,
                        model,
                        class_index=predicted_index,
                    )

                    overlay = overlay_heatmap(
                        image,
                        heatmap,
                    )


                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(11, 4.5)
                )

                axes[0].imshow(
                    image.resize(IMAGE_SIZE)
                )

                axes[0].set_title(
                    "Original MRI",
                    fontsize=13,
                    fontweight="bold",
                )

                axes[0].axis("off")


                axes[1].imshow(
                    overlay
                )

                axes[1].set_title(
                    "Grad-CAM Explanation",
                    fontsize=13,
                    fontweight="bold",
                )

                axes[1].axis("off")


                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )


                # =================================================
                # DOWNLOAD GRAD-CAM IMAGE
                # =================================================

                gradcam_buffer = io.BytesIO()

                fig.savefig(
                    gradcam_buffer,
                    format="png",
                    bbox_inches="tight",
                    dpi=200,
                )

                gradcam_buffer.seek(0)

                st.download_button(
                    label="🔥 Download Grad-CAM Image",
                    data=gradcam_buffer,
                    file_name="alzheimer_gradcam_explanation.png",
                    mime="image/png",
                    use_container_width=True,
                )

                plt.close(fig)


            except Exception as exc:

                st.warning(
                    f"Grad-CAM could not be generated: {exc}"
                )


            # =================================================
            # DOWNLOAD PDF REPORT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📑 Professional PDF Report'
                '</div>',
                unsafe_allow_html=True,
            )

            try:
                pdf_data = generate_pdf_report(
                    image=image,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    class_names=class_names,
                    probabilities=probabilities,
                    gradcam_png=gradcam_buffer,
                )

                st.download_button(
                    label="📑 Download Complete PDF Report",
                    data=pdf_data,
                    file_name="alzheimer_mri_complete_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                st.caption(
                    "PDF includes MRI image, prediction, confidence, class probabilities, "
                    "Grad-CAM explanation and disclaimer."
                )
            except Exception as exc:
                st.error(f"PDF report could not be generated: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>Alzheimer's Disease MRI Classifier</b><br>

    Deep Learning • EfficientNet-B0 • Transfer Learning • Grad-CAM<br><br>

    ⚠️ For educational and research purposes only.
    This system does not provide medical diagnosis.

    </div>
    """,
    unsafe_allow_html=True,
)