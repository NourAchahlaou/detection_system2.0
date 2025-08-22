Real-Time Piece Verification & Identification System
🛠 Project Overview
This project is a real-time industrial piece identification and verification system designed to automate quality control and validation tasks for Airbus. It leverages advanced object detection (YOLO), GPU-optimized training, and a modern microservice-based architecture to ensure accurate, traceable, and user-friendly inspection workflows.

📌 Features
✅ Detection & Verification
YOLOv8-based object detection to classify and locate industrial parts.
 
Label consistency checking between physical and predicted labels.

Misplaced piece detection and real-time validation in a lot.

Confidence scores and detection statistics shown clearly in UI.

📂 Dataset Management
Add, delete, and manage piece images (min. 10 required per piece).

Piece duplication checks using smart label comparison.

Airbus dataset integration for fetching official documents.

🖼 Annotation System
Intelligent contouring using AI for accurate bounding boxes.

Multiple bounding box support, perpendicular line tools.

Full annotation lifecycle: annotate → verify → save.

📷 Image Capture
Auto-capture at lot entry and after corrections.

Structured screenshot storage and visual comparison.

Minimum image requirements enforced per piece.

📊 Dashboard & Monitoring
Live training statistics: loss, accuracy, resource usage.

System health monitoring: GPU, CPU, RAM, disk via NVML & psutil.

Notification center for overload, errors, new piece detection.

🧠 Training System
GPU-optimized with:

Auto-batch size tuning.

Mixed precision support.

Hyperparameter fine-tuning.

Stress testing on large datasets.

Automatic alerts when overload is detected.

🧩 Microservices Architecture
Modularized using NestJS, Docker, Anaconda.

Clear separation between Detection, Annotation, Monitoring, User Management, and Training.

👤 User Management
Admin dashboard for full control.

Roles: Technicians (full access), Auditors (read-only).

Activity logs, shift times, task tracking, and performance reports.

Real-time alerts for profile updates, shift conflicts, and piece detection events.

🧱 Tech Stack
Frontend: React

Backend: FastAPI  (microservices)

Model: YOLOv8

Database: PostgreSQL

Containerization: Docker

Monitoring: psutil, NVIDIA NVML

Notifications: Firebase Cloud Messaging //

Annotation UI: Custom React components with advanced drawing tools

🔄 Project Workflow
Detection: Pieces pass through the detection system, identifying misplaced or unknown items.

Validation: Labels and expected lot counts are verified.

Annotation: New pieces are annotated using AI tools.

Database Update: New pieces are registered and added to missions.

Monitoring: All system and training activities are tracked via the dashboard.

🚀 Getting Started
Prerequisites
Python 3.10+

Node.js 16+

Docker & Docker Compose

Anaconda

NVIDIA drivers + CUDA for GPU support

Setup
Clone the repository


git clone https://github.com/NourAchahlaou/detection_system2.0
cd detection_system2.0
Start services using Docker


docker-compose up --build
Train YOLO model (optional)


yolo train model=yolov8n.pt data=your_data.yaml --img 640 --batch -1 --epochs 100 --amp
📁 Folder Structure (Recommended)

📦 piece-verification-system
 ┣ 📂 backend
 ┃ ┣ 📂 detection-service
 ┃ ┣ 📂 annotation-service
 ┃ ┣ 📂 monitoring-service
 ┃ ┣ 📂 user-service
 ┃ ┣ 📜 main.py
 ┣ 📂 frontend
 ┃ ┣ 📂 src
 ┃ ┃ ┣ 📂 components
 ┃ ┃ ┣ 📂 pages
 ┃ ┃ ┣ 📜 App.tsx
 ┣ 📂 yolo-model
 ┣ 📂 docker
 ┣ 📜 docker-compose.yml
 ┣ 📜 README.md
📅 Roadmap
Phase 1: Stabilization
 Finalize detection microservice

 Tune YOLO hyperparameters

 Fix login bugs and improve logging

Phase 2: Architecture Redesign
 Transition to microservices with Docker/NestJS

 Document architecture & class diagrams

 Enable cross-service communication

Phase 3: Dashboard Completion
 Add training graphs

 Real-time system performance updates

 Integrated notifications

📣 Contributing
We welcome contributions! Please fork the repo, open issues, and submit PRs.

🧑‍💼 Author & Supervision
👤 Presented by: Achahlaou Nour
✈️ Organization: Airbus



📜 License
This project is proprietary and under Airbus internal use. Please contact the project supervisor for access and collaboration permissions.