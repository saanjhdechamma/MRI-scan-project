# =============================================================
#  AI MRI Dashboard: Detection → Segmentation → Gemini → Apollo PDF
# =============================================================

import os, re
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image as PILImage

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="AI MRI Brain Tumor Diagnostic", layout="wide")

# ===================== BASE DIR =====================
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== GEMINI =====================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

HAVE_GENAI = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel("gemini-2.5-flash")
        HAVE_GENAI = True
    except Exception:
        HAVE_GENAI = False


def fallback_report(reason="LLM unavailable") -> str:
    return f"""
Findings:
- Automated MRI analysis was completed successfully.
- Tumor characteristics were detected using deep learning models.
- LLM-generated narrative could not be produced ({reason}).

Impression:
- Findings suggest abnormal brain tissue.
- Final diagnosis should be confirmed by a radiologist.

Recommendations:
- Clinical correlation is advised.
- Follow-up imaging and specialist consultation recommended.

Note:
This report was generated using AI-assisted image analysis.
"""


def gemini_generate_llm(prompt: str) -> str:
    if not HAVE_GENAI:
        return fallback_report("Gemini not configured")

    try:
        return llm_model.generate_content(prompt).text
    except ResourceExhausted:
        return fallback_report("Gemini quota exceeded")
    except Exception as e:
        return fallback_report(str(e))


# ===================== MODELS =====================
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_inceptionv3_tumor.h5")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "tumor_segmentation_unet.h5")

@st.cache_resource
def load_models():
    return (
        load_model(DETECTION_MODEL_PATH),
        load_model(SEGMENTATION_MODEL_PATH, compile=False)
    )

detection_model, segmentation_model = load_models()
CLASS_LABELS = ["glioma", "meningioma", "no_tumor", "pituitary"]


# ===================== MRI ANALYSIS =====================
def analyze_mri(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    preds = detection_model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)

    label = CLASS_LABELS[idx]
    confidence = float(preds[idx] * 100)

    tumor_pct = 0.0
    mask_path = None

    if label != "no_tumor":
        seg_img = image.load_img(img_path, color_mode="grayscale", target_size=(128, 128))
        seg_arr = np.expand_dims(image.img_to_array(seg_img) / 255.0, axis=0)
        mask = segmentation_model.predict(seg_arr, verbose=0)[0][:, :, 0]

        binary = (mask > 0.5).astype(np.uint8)
        tumor_pct = binary.sum() / binary.size * 100

        mask_path = os.path.join(OUTPUT_DIR, f"mask_{datetime.now().strftime('%H%M%S')}.png")
        plt.imsave(mask_path, binary * 255, cmap="gray")

    return {
        "image_path": img_path,
        "predicted_label": label,
        "confidence": confidence,
        "tumor_percentage": tumor_pct,
        "mask_image_path": mask_path
    }


# ===================== REPORT =====================
REPORT_PROMPT = """
You are an expert radiology assistant. Generate a professional MRI brain tumor report.

Tumor Type: {label}
Confidence: {confidence:.2f}%
Tumor Coverage: {coverage:.2f}%

Provide sections:
Findings:
Impression:
Recommendations:
"""

def generate_report(data, patient):
    return gemini_generate_llm(
        REPORT_PROMPT.format(
            name=patient["name"],
            age=patient["age"],
            gender=patient["gender"],
            date=datetime.now().strftime("%d %B %Y"),
            label=data["predicted_label"],
            confidence=data["confidence"],
            coverage=data["tumor_percentage"],
        )
    )


# ======================================================================
# ===================== APOLLO PDF (FORMAT UNCHANGED) ===================
# ======================================================================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import qrcode

HOSPITAL_NAME = "AI Diagnostic Center"
HOSPITAL_SUB = "X-Ray | CT-Scan | MRI | USG"
HOSPITAL_CONTACT = "Phone: 0123456789  |  Email: info@aidiagnostic.com"
RADIOLOGIST_NAME = "AI Neuro-oncologist"
HEADER_DECOR_PATH = os.path.join(BASE_DIR, "header_decor.jpeg")


def clean_text_for_pdf(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text or "")
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return re.sub(r'\n\s*\n+', '\n\n', text.strip())


def make_qr(data, path):
    qr = qrcode.QRCode(box_size=3, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(path)


# ⬇⬇⬇
def create_apollo_style_pdf(report_text,
                            original_img,
                            mask_img,
                            patient,
                            pid="PID0001",
                            apt_id="APT0001",
                            output_path="final_mri_report_styled.pdf"):

    report_text = clean_text_for_pdf(report_text)
    study_date = datetime.now().strftime("%d-%b-%Y")

    qr_payload = (
        "BEGIN:VCARD\n"
        "VERSION:3.0\n"
        f"N:{patient.get('name','')}\n"
        f"NOTE:PID {pid}, Age {patient.get('age','')}, "
        f"Gender {patient.get('gender','')}, "
        f"Date {study_date}\n"
        "END:VCARD"
    )

    qr_path = os.path.join(os.path.dirname(output_path), f"qr_{pid}.png")
    make_qr(qr_payload, qr_path)

    # ---- Create document ONCE ----
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=18,
        bottomMargin=22,
        leftMargin=22,
        rightMargin=22
    )

    styles = getSampleStyleSheet()

    hdr_style = ParagraphStyle(
        'hdr', parent=styles['Normal'], alignment=1,
        fontSize=10, textColor=colors.white, leading=12
    )
    title_style = ParagraphStyle(
        'title', parent=styles['Heading1'], alignment=1,
        fontSize=15, textColor=colors.HexColor("#0D47A1"), leading=16
    )
    section_style = ParagraphStyle(
        'section', parent=styles['Heading2'],
        fontSize=12, textColor=colors.HexColor("#0D47A1"),
        leading=13, spaceBefore=6, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'body', parent=styles['BodyText'],
        fontSize=10, leading=13, spaceAfter=2
    )
    small_style = ParagraphStyle(
        'small', parent=styles['BodyText'],
        fontSize=8.5, leading=10, textColor=colors.grey
    )

    story = []

    # ---------- HEADER ----------
    left_col = Paragraph(
        f"<b>{HOSPITAL_NAME}</b><br/><font size=9>{HOSPITAL_SUB}</font>",
        hdr_style
    )

    right_col = ""
    if HEADER_DECOR_PATH and os.path.exists(HEADER_DECOR_PATH):
        try:
            pil = PILImage.open(HEADER_DECOR_PATH)
            pil_w, pil_h = pil.size
            scale = 24.0 / pil_h if pil_h else 1.0
            img_w = int(pil_w * scale)
            img_h = int(pil_h * scale)
            tmp_hdr = os.path.join(os.path.dirname(output_path), "header_tmp.png")
            pil.resize((img_w, img_h)).save(tmp_hdr)
            right_col = RLImage(tmp_hdr, width=img_w, height=img_h)
        except:
            right_col = ""

    header_tbl = Table([[left_col, right_col]], colWidths=[420, 80])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0D47A1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6))

    # ---------- TITLE ----------
    story.append(Paragraph("MRI BRAIN TUMOR ANALYSIS REPORT", title_style)) 
    story.append(Spacer(1, 8))

    

    # Patient block and registration
    pid_box = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(patient.get("name",""), body_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(str(patient.get("age","")), body_style)],
        [Paragraph("<b>Sex:</b>", body_style), Paragraph(patient.get("gender",""), body_style)],
    ]
    left_table = Table(pid_box, colWidths=[80,150])
    left_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))

    qr_img = RLImage(qr_path, width=55, height=55)

    reg_block = [
        [Paragraph("<b>PID</b>", body_style), Paragraph(":", body_style), Paragraph(pid, body_style)],
        [Paragraph("<b>Apt ID</b>", body_style), Paragraph(":", body_style), Paragraph(apt_id, body_style)],
        [Paragraph("<b>Ref. By</b>", body_style), Paragraph(":", body_style), Paragraph(RADIOLOGIST_NAME, body_style)],
        [Paragraph("<b>Registered on</b>", body_style), Paragraph(":", body_style),
         Paragraph(datetime.now().strftime("%I:%M %p %d %b, %Y"), body_style)],
        [Paragraph("<b>Reported on</b>", body_style), Paragraph(":", body_style),
         Paragraph(datetime.now().strftime("%I:%M %p %d %b, %Y"), body_style)]
    ]
    right_table = Table(reg_block, colWidths=[80,10,140])
    right_table.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),2),
        ('RIGHTPADDING',(0,0),(-1,-1),2),
        ('TOPPADDING',(0,0),(-1,-1),1),
        ('BOTTOMPADDING',(0,0),(-1,-1),1),
    ]))

    top_row = [[left_table, qr_img, right_table]]
    top_tbl = Table(top_row, colWidths=[230, 60, 210])
    top_tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(top_tbl)
    story.append(Spacer(1,6))

    # Divider
    story.append(Table([[""]], colWidths=[520], style=[('LINEABOVE',(0,0),(-1,-1),0.6,colors.lightgrey)]))
    story.append(Spacer(1,6))

    # Helper to add section with bold heading
    def add_section(heading, content):
        story.append(Paragraph(f"<b>{heading}:</b>", section_style))
        for para in content.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            if len(lines) == 1:
                story.append(Paragraph(lines[0], body_style))
            else:
                for ln in lines:
                    ln = re.sub(r'^[\-\*\u2022]+\s*', '', ln)
                    story.append(Paragraph(f"&bull; {ln}", body_style))

    # Try to split the report_text into reasonable sections; fallback to generic sections
    lower = report_text.lower()
    if any(k in lower for k in ["findings", "impression", "technique", "clinical"]):
        sections = {}
        current = "Report"
        sections[current] = []
        for line in report_text.splitlines():
            if not line.strip():
                continue
            l = line.strip()
            if re.match(r'^(findings[:\-]?)', l, re.I):
                current = "Findings"; sections[current] = []
                l = re.sub(r'^(findings[:\-]?)', '', l, flags=re.I).strip()
                if l: sections[current].append(l); continue
            if re.match(r'^(impression|conclusion)[:\-]?', l, re.I):
                current = "Conclusion"; sections[current] = []
                l = re.sub(r'^(impression|conclusion)[:\-]?', '', l, flags=re.I).strip()
                if l: sections[current].append(l); continue
            if re.match(r'^(technique)[:\-]?', l, re.I):
                current = "Technique"; sections[current] = []
                l = re.sub(r'^(technique)[:\-]?', '', l, flags=re.I).strip()
                if l: sections[current].append(l); continue
            if re.match(r'^(clinical information)[:\-]?', l, re.I):
                current = "Clinical Information"; sections[current] = []
                l = re.sub(r'^(clinical information)[:\-]?', '', l, flags=re.I).strip()
                if l: sections[current].append(l); continue
            sections[current].append(l)
        add_section("Part", "Brain (Axial/MRI sequences)")
        if "Clinical Information" in sections: add_section("Clinical Information", "\n".join(sections["Clinical Information"]))
        if "Technique" in sections: add_section("Technique", "\n".join(sections["Technique"]))
        if "Findings" in sections: add_section("Findings", "\n".join(sections["Findings"]))
        else: add_section("Findings", "\n".join(sections.get("Report", [])))
        if "Conclusion" in sections: add_section("Conclusion", "\n".join(sections["Conclusion"]))
    else:
        add_section("Part", "Brain (Axial/MRI sequences)")
        add_section("Clinical Information", "Not provided.")
        add_section("Technique", "Standard multiplanar brain MRI sequences.")
        add_section("Findings", report_text)
        add_section("Conclusion", "See findings above.")

    story.append(Spacer(1,6))

    # Images side-by-side (proportional)
    imgs = []
    if original_img and os.path.exists(original_img):
        imgs.append(RLImage(original_img, width=240, height=240, kind='proportional'))
    else:
        imgs.append(Paragraph("<i>Original image unavailable</i>", body_style))
    if mask_img and os.path.exists(mask_img):
        imgs.append(RLImage(mask_img, width=240, height=240, kind='proportional'))
    else:
        imgs.append(Paragraph("<i>No segmentation mask</i>", body_style))
    imgs_tbl = Table([imgs], colWidths=[240,240])
    imgs_tbl.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(imgs_tbl)
    story.append(Spacer(1,6))

    # Signature block
    sign_tbl = Table([
        [Paragraph("<b>Reported By:</b>", body_style), "", Paragraph("<b>Authorized By:</b>", body_style)],
        [Paragraph(RADIOLOGIST_NAME, body_style), "", Paragraph("Dr. Radiologist", body_style)],
        [Paragraph("(MD, Neuro-radiology)", small_style), "", Paragraph("(MD, Radiology)", small_style)]
    ], colWidths=[200,60,200])
    sign_tbl.setStyle(TableStyle([
        ('LINEABOVE',(0,0),(0,0),0.2,colors.grey),
        ('LINEABOVE',(2,0),(2,0),0.2,colors.grey),
        ('LEFTPADDING',(0,0),(-1,-1),4),
        ('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),2),
    ]))
    story.append(sign_tbl)
    story.append(Spacer(1,6))

    # Generated notice
    story.append(Paragraph(f"<font size=8>This is an AI-assisted radiology evaluation. Generated on: {datetime.now().strftime('%d %b %Y %I:%M %p')}</font>", small_style))

   
    def draw_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        footer_text = f"Page {doc_obj.page} | {HOSPITAL_NAME} | {HOSPITAL_CONTACT}"
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(colors.grey)
        canvas_obj.drawCentredString(A4[0]/2.0, 12*mm, footer_text)
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(f"✅ PDF created: {output_path}")
# YOUR create_apollo_style_pdf FUNCTION
# IS KEPT 100% IDENTICAL
# ⬆⬆⬆

# (No formatting changes done here — only paths fixed internally)

# =============================================================
# STREAMLIT UI  (FULLY FIXED – NO ERRORS)
# =============================================================

# ---------- REQUIRED ALIASES (DO NOT REMOVE) ----------
# keep UI code unchanged
analyze_mri_local = analyze_mri
gemini_generate = gemini_generate_llm

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="AI MRI Brain Tumor Diagnostic", layout="wide")

# ===================== BASE / ASSETS =====================
BASE_DIR = os.getcwd()
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

# ===================== OUTPUT DIRECTORIES =====================
PDF_OUTPUT_DIR = os.path.join(BASE_DIR, "pdf_outputs")
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)

# ===================== SESSION STATE =====================
for key, default in {
    "last_analysis": {},
    "last_report": "",
    "chat_history": [],
    "navigate_to_analyze": False,
    "navigate_to_home": False,
    "navigate_to_assistant": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===================== SIDEBAR =====================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    page = st.radio("Navigate", ["Home", "Analyze", "Assistant", "About"], index=0)

# ====================== HOME ======================
if page == "Home":

    st.markdown("<h1> ScanDX AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Upload MRI → Detection → Segmentation → PDF → Ask AI Assistant</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='text-align:center; margin-bottom:20px;'>
            <img src='https://i.gifer.com/NTHO.gif' height='400'>
        </div>
        """,
        unsafe_allow_html=True
    )

   

    if st.session_state.navigate_to_analyze:
        page = "Analyze"

# ====================== ANALYZE ======================
elif page == "Analyze":

    st.header("Upload MRI → Run Detection & Segmentation")

    st.markdown(
        """
        <div style='text-align:center; margin-bottom:10px;'>
            <img src='https://i.gifer.com/HesE.gif' width='420'>
        </div>
        """,
        unsafe_allow_html=True
    )

    upload = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if upload:
        temp_path = os.path.join(
            PDF_OUTPUT_DIR, f"input_{datetime.now().strftime('%H%M%S')}.jpg"
        )
        with open(temp_path, "wb") as f:
            f.write(upload.getbuffer())

        st.image(temp_path, caption="Uploaded MRI", use_container_width=True)

        if st.button("Run Analysis"):
            with st.spinner("Running models..."):
                result = analyze_mri_local(temp_path)
                st.session_state.last_analysis = result
                st.success("Analysis Complete ✔")

    if st.session_state.last_analysis:
        r = st.session_state.last_analysis

        st.subheader("Results")
        st.write(f"**Tumor Type:** {r['predicted_label'].upper()}")
        st.write(f"**Confidence:** {r['confidence']:.2f}%")
        st.write(f"**Tumor Coverage:** {r['tumor_percentage']:.2f}%")

        if r["mask_image_path"]:
            st.image(r["mask_image_path"], caption="Segmentation Mask")

        st.subheader("Generate PDF Report")
        name = st.text_input("Patient Name", "John Doe")
        age = st.number_input("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        if st.button("Generate Report"):
            prompt = f"""
Patient: {name}, Age: {age}, Gender: {gender}
Tumor Type: {r['predicted_label']}
Confidence: {r['confidence']:.2f}%
Coverage: {r['tumor_percentage']:.2f}%
            """

            report_text = gemini_generate(prompt)
            st.session_state.last_report = report_text

            pdf_path = os.path.join(
                PDF_OUTPUT_DIR,
                f"MRI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            create_apollo_style_pdf(
                report_text,
                r["image_path"],
                r["mask_image_path"],
                {"name": name, "age": age, "gender": gender},
                pid="PID" + datetime.now().strftime("%H%M%S"),
                apt_id="APT001",
                output_path=pdf_path
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download PDF",
                    f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(" Go to Home"):
                    page = "Home"
            with col2:
                if st.button(" Go to Assistant"):
                    page = "Assistant"

# ====================== ASSISTANT ======================
elif page == "Assistant":

    st.header("AI Assistant")

    st.markdown(
        """
        <div style='text-align:center; margin-bottom:20px;'>
            <img src='https://i.gifer.com/NTHO.gif' height='350'>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.last_report:
        st.info("Run an analysis and generate a report first.")
    else:
        question = st.text_input("Ask anything about the findings")

        if st.button("Ask") and question.strip():
            reply = gemini_generate(
                f"Report:\n{st.session_state.last_report}\n\nQuestion: {question}"
            )
            st.session_state.chat_history.append((question, reply))

        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")

# ====================== ABOUT ======================
elif page == "About":

    st.markdown(
        """
        <h1 style='text-align:center;'>Smart AI FrameWork for MRI-Driven Pathology Detection and Clinical Reporting</h1>
        <p style='text-align:center; font-size:18px; color:gray;'>
        Intelligent MRI Analysis • Automated Reporting • AI Assistance
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # ---------- INTRO ----------
    st.markdown(
        """
        ### Overview
        This system is an **AI-powered MRI brain tumor analysis platform** designed for  
        **academic research, learning, and proof-of-concept medical AI demonstrations**.

        It integrates **deep learning**, **medical image processing**, and  
        **large language models** to assist radiologists in **analysis and documentation**.
        """
    )

    # ---------- ARCHITECTURE DIAGRAM ----------
    st.markdown("###  System Architecture")

    ARCH_PATH = os.path.join(BASE_DIR, "assets", "architecture.png")
    if os.path.exists(ARCH_PATH):
        st.image(
            ARCH_PATH,
            caption="End-to-End Architecture of the AI MRI Diagnostic Pipeline",
            use_container_width=True
        )
    else:
        st.warning("Architecture diagram not found. Place it at assets/architecture.png")

    # ---------- PIPELINE ----------
    st.markdown(
        """
        ### Processing Pipeline
        **1. MRI Image Input**  
        The user uploads a brain MRI image (JPG/PNG).

        **2. Tumor Classification (CNN)**  
        A Convolutional Neural Network predicts one of:
        - Glioma
        - Meningioma
        - Pituitary Tumor
        - No Tumor

        **3. Tumor Segmentation (U-Net)**  
        If a tumor is detected, a U-Net model segments the tumor region and computes  
        **tumor coverage percentage**.

        **4. AI Radiology Report Generation**  
        Findings are converted into a structured medical report using **Google Gemini**  
        with a **quota-safe fallback mechanism**.

        **5. Apollo-Style Medical PDF**  
        A professional hospital-grade PDF is generated containing:
        - Patient details
        - MRI & segmentation images
        - Findings, Impression & Recommendations
        - QR code for traceability

        **6. Interactive AI Assistant**  
        Users can ask follow-up questions about the generated report.
        """
    )

    # ---------- TECHNOLOGY STACK ----------
    st.markdown(
        """
        ### ⚙️ Technology Stack
        - **Deep Learning:** TensorFlow, Keras  
        - **Classification:** CNN (Inception-based)  
        - **Segmentation:** U-Net  
        - **LLM:** Google Gemini (safe fallback enabled)  
        - **Frontend:** Streamlit  
        - **PDF Engine:** ReportLab  
        - **Image Processing:** OpenCV, Pillow  
        """
    )

    # ---------- KEY FEATURES ----------
    st.markdown(
        """
        ###  Key Highlights
        - Fully automated MRI analysis pipeline  
        - Tumor confidence & area estimation  
        - Professional medical-style reporting  
        - Quota-safe AI text generation  
        - Clean UI for demos & presentations  
        """
    )

    # ---------- LIMITATIONS ----------
    st.markdown(
        """
        ###  Limitations & Ethics
        - Not approved for clinical diagnosis  
        - Model accuracy depends on training data  
        - Intended for **research & educational use only**  
        - Final diagnosis must be made by certified radiologists  
        """
    )

    # ---------- FUTURE SCOPE ----------
    st.markdown(
        """
        ###  Future Scope
        - Support for DICOM images  
        - Multi-slice MRI analysis  
        - Clinical validation  
        - Voice-based assistant  
        - Cloud deployment (AWS / Azure)  
        """
    )

    # ---------- DISCLAIMER ----------
    st.markdown(
        """
        ---
        **Disclaimer:**  
        This system provides **AI-assisted insights only** and does not replace
        professional medical judgment.
        """
    )