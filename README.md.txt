# TruthLensAI 🧠🔍

TruthLensAI is a full-stack AI system that detects:
- Fake News using NLP (text classification)
- Deepfake Images using CNN (MobileNetV2)

## 🚀 Features
- Fake news detection from text
- Deepfake image detection
- FastAPI backend
- Modern frontend UI
- Swagger API documentation

## 🛠 Tech Stack
- Python, FastAPI
- TensorFlow, OpenCV
- HTML, CSS, JavaScript

## 📂 Project Structure
TruthLensAI/
├── backend/
├── frontend/
├── ml/


## ▶️ How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Use VS Code Live Server or:

cd frontend
python -m http.server 5500
http://localhost:5500/index.html

📄 API Docs
http://127.0.0.1:8000/docs


## 🔮 Future Improvements

- Use larger deepfake datasets (DFDC, Celeb-DF)
- Improve CNN accuracy
- Add video deepfake detection
- Deploy system on cloud
