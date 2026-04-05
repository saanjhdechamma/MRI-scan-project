


#  ScanDX AI – MRI Brain Tumor Analysis System

**ScanDX AI** is an AI-based MRI brain tumor analysis system built to understand how **machine learning, medical imaging, and cloud deployment** work together in a real-world application.

The system allows users to upload an MRI brain image, automatically detect and segment brain tumors, generate a professional radiology-style PDF report, and interact with an AI assistant.

**This project is for academic and learning purposes only. It is NOT a medical diagnostic tool.**

---

## What This Project Does

- Upload an MRI brain image  
- Detect whether a tumor is present  
- Identify the tumor type  
- Segment the tumor region  
- Calculate tumor coverage percentage  
- Generate a radiology-style report  
- Export a hospital-style PDF  
- Ask follow-up questions using an AI assistant  
- Fully deployed and running on **AWS Cloud**

##  Technologies Used

###  Frontend
- **Streamlit** – Web UI framework

###  Machine Learning
- **CNN (InceptionV3)** – Tumor classification  
- **U-Net** – Tumor segmentation  
- **TensorFlow / Keras**  
- **NumPy, OpenCV**

###  AI (LLM)
- **Google Gemini**
- Used only for:
  - Report text generation
  - AI assistant
- Quota-safe fallback implemented

###  PDF Generation
- **ReportLab**
- Apollo-style medical report layout
- QR code included

###  Cloud & DevOps
- **Docker**
- **Amazon ECR**
- **Amazon ECS (Fargate)**
- **Application Load Balancer**
- **AWS IAM**

### Architecture Overview
![Architecture](assets/architecture.png)

##  Project Structure

```text
MRI_Brain/
│
├── app.py                     # Main Streamlit application
├── models/
│   ├── best_inceptionv3_tumor.h5
│   └── tumor_segmentation_unet.h5
│
├── assets/
│   ├── logo.png
│   └── architecture.png
│
├── pdf_outputs/               # Generated reports
│
├── Dockerfile
├── requirements.txt
└── README.md

How the System Works (Simple Flow)
	1.	User uploads an MRI image
	2.	Image is preprocessed (resize + normalize)
	3.	CNN model predicts tumor type
	4.	U-Net model segments tumor area
	5.	Tumor coverage is calculated
	6.	AI generates a medical report
	7.	PDF report is created
	8.	User can ask questions using AI assistant

Report Design Logic
	•	Patient details appear only in the header
	•	Findings section contains ONLY imaging observations
	•	No patient name, age, or gender inside findings
	•	Follows proper radiology reporting standards

Example

Findings:
	•	Well-defined mass lesion observed
	•	Hyperintense signal in affected region
	•	Mild mass effect noted

☁️ AWS Deployment Summary
	•	Docker image built for linux/amd64
	•	Image pushed to Amazon ECR
	•	Service deployed on Amazon ECS (Fargate)
	•	Application exposed using Application Load Balancer
	•	Updates performed using Force New Deployment



🐳 Important Docker Note (Mac Users)

Since this project was built on a Mac (ARM architecture), the Docker image is built using:

docker buildx build --platform linux/amd64 .

This avoids ECS errors such as:

CannotPullContainerError: no matching platform

Limitations
	•	Not approved for clinical use
	•	Accuracy depends on training data
	•	No DICOM support (only image files)
	•	Gemini API has quota limits
	•	No user authentication

 Future Improvements
	•	DICOM file support
	•	Multi-sequence MRI analysis
	•	User login and report history
	•	Radiologist feedback system
	•	CI/CD using GitHub Actions
	•	Auto-scaling on AWS

 Purpose of This Project
	•	Learn medical image processing
	•	Apply deep learning models
	•	Integrate LLMs with ML systems
	•	Deploy a full-stack ML application on AWS
	•	Showcase AI + Cloud + DevOps skills


👤 Author

Prajwal S
Engineering Student
Interested in AI, ML, Cloud & DevOps

 Disclaimer

This system is created only for educational and demonstration purposes.
Always consult a qualified medical professional for real diagnosis.




