# Spam Mail Detection Pipeline

## 🚀 Overview
The **Spam Mail Detection Pipeline** is a reproducible workflow for detecting spam emails. The repository contains Jupyter Notebooks for data exploration and model training, a lightweight inference app, and Docker configurations to run the app in a consistent environment.

## 🛠 Tech Stack
- **Notebooks:** Jupyter Notebook
- **Language:** Python (pandas, scikit-learn, numpy)
- **Modeling:** scikit-learn (Logistic Regression, Naive Bayes, etc.)
- **Serialization:** joblib / pickle
- **Deployment:** Docker, docker-compose (optional)

> Note: Deployment is optional — you can run the notebooks and app locally without Docker. See the "Getting Started" section for local setup.

## ✨ Features
- End-to-end training pipeline in notebooks (data cleaning, vectorization, modeling, evaluation)  
- Model serialization for inference (joblib / pickle)  
- Simple inference API / app for text prediction  
- Docker compose for reproducible environment and quick deployment (optional)

## 📂 Project Structure
```
/ (root)
├── "Spam Mail Detection Training Pipeline"/   # Jupyter notebooks for data exploration and model training
├── app/                                      # Inference API or web app (model serving)
├── docker-compose.yml                        # Docker compose to build/run app and services
```

Note: I inspected the repository and will keep the top-level folders and files intact when adding this README.

## 🚀 Getting Started

### Prerequisites
Ensure you have:
- Python 3.8+ (recommended)
- pip
- Git
- (Optional) Docker & Docker Compose

### Installation
Clone the repository:
```sh
git clone https://github.com/DevSharma03/Spam_Mail_Detection_Pipeline.git
cd Spam_Mail_Detection_Pipeline
```

(Recommended) Create and activate a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

Install dependencies (check `app/requirements.txt` if present):
```sh
pip install -r app/requirements.txt  # if present
# or
pip install jupyter pandas scikit-learn numpy matplotlib seaborn joblib
```

### Running Notebooks
Start Jupyter and open the notebooks inside the "Spam Mail Detection Training Pipeline" folder:
```sh
jupyter notebook
# or
jupyter lab
```
Run the notebooks in the suggested order to reproduce preprocessing, training, and evaluation.

### Running the App (Inference)
1. Inspect the `app/` directory for an entrypoint (e.g., `app.py`, `server.py`).  
2. Install app dependencies and run the server:
```sh
cd app
pip install -r requirements.txt  # if present
python app.py  # or the module that starts the server
```
3. Test prediction endpoint (adjust host/port as required):
```sh
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text":"Free money!!!"}'
```

## 🛠 Deployment (Optional)
If you prefer containers, use Docker Compose (if configured):
```sh
docker-compose up --build
```

Tips:
- Ensure model artifacts (serialized model + vectorizer) are available to the container or included in the image build.  
- Set required environment variables inside the compose file or the hosting environment.

## 🔍 Troubleshooting & Issues Faced
During development we encountered several common issues — symptoms and fixes that can help:

- Notebook kernel or missing package errors
  - Ensure your virtual environment has the correct packages installed and the notebook kernel uses it.

- Missing `requirements.txt` or dependency mismatches
  - Check `app/` for a `requirements.txt`. If missing, install core ML packages or pin versions and add a `requirements.txt`.

- MongoDB / external service connection issues (if used)
  - Verify connection strings, credentials, and network access (firewalls/VPC rules).

- Model serialization / path errors
  - Confirm serialized model and preprocessor (e.g., TF-IDF vectorizer) paths used by the app match the repository layout.

- Docker build failures
  - Verify Dockerfile paths, existence of model artifacts, and that COPY commands reference correct relative paths.

- Inference returns unexpected predictions
  - Ensure the same preprocessing pipeline used during training is applied at inference time.

- CORS or network issues when calling the app from a browser
  - Configure CORS headers in the app or use a proxy to avoid cross-origin issues.

If you encounter an issue not listed here, please open an issue with logs and steps to reproduce.

## Reproducibility tips
- Pin package versions in `requirements.txt` to reproduce notebook runs.  
- Save model artifacts (model + vectorizer/preprocessor) in a `models/` directory and update the app to load them from that path.  
- Document the notebook execution order in a small README inside the "Spam Mail Detection Training Pipeline" folder.

## 🤝 Contributing
Contributions are welcome! Please open issues or submit PRs. When opening a PR, include:
- A clear description of the change  
- Steps to reproduce (if bugfix)  
- Any notebook, script, or model artifacts added or changed

## 📜 License
This project does not currently include a license file. Add a license (e.g., MIT) if you want to make the project explicitly open-source.

## 📞 Contact
- **Repo owner:** DevSharma03
