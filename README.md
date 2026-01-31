# End-to-End ML Pipeline with CI/CD

A complete machine learning pipeline demonstrating data processing, model training, REST API deployment, and automated CI/CD using Python, FastAPI, Docker, and GitHub Actions.

## 🎯 Project Overview

This project implements a full ML pipeline for **Iris flower classification** with the following components:

- **Data Ingestion**: Load and validate data with train/test splitting
- **Feature Engineering**: Standardization and preprocessing
- **Model Training**: Random Forest classifier with cross-validation
- **Model Evaluation**: Comprehensive metrics and reporting
- **REST API**: FastAPI service for predictions
- **Testing**: Comprehensive unit tests with pytest
- **Docker**: Containerized deployment
- **CI/CD**: Automated pipeline with GitHub Actions

## 📁 Project Structure

```
end-to-end-ml-cicd/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Data loading and splitting
│   ├── feature_engineering.py   # Feature transformations
│   ├── model_training.py        # Model training logic
│   ├── model_evaluation.py      # Metrics and evaluation
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI application
│       ├── models.py            # Pydantic models
│       └── prediction.py        # Prediction service
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions workflow
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pytest.ini
├── .flake8
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd end-to-end-ml-cicd
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🔧 Running the ML Pipeline

### Step 1: Data Ingestion
```bash
python src/data_ingestion.py
```
This will:
- Load the Iris dataset
- Validate data quality
- Split into train/test sets (80/20)
- Save processed data to `data/` directory

### Step 2: Feature Engineering
```bash
python src/feature_engineering.py
```
This will:
- Apply StandardScaler transformation
- Save feature transformer to `models/`

### Step 3: Model Training
```bash
python src/model_training.py
```
This will:
- Perform 5-fold cross-validation
- Train Random Forest classifier
- Display feature importance
- Save trained model to `models/`

### Step 4: Model Evaluation
```bash
python src/model_evaluation.py
```
This will:
- Calculate performance metrics
- Generate confusion matrix
- Create classification report
- Save results to `results/evaluation_results.json`

## 🌐 Running the API

### Local Development

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "confidence": 0.95,
  "probabilities": [0.95, 0.03, 0.02]
}
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run specific test file:
```bash
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

### Build the Docker image

First, ensure you've run the ML pipeline to generate model artifacts:
```bash
python src/data_ingestion.py
python src/feature_engineering.py
python src/model_training.py
```

Build the image:
```bash
docker build -t ml-pipeline:latest .
```

### Run the container
```bash
docker run -d -p 8000:8000 --name ml-api ml-pipeline:latest
```

### Test the containerized API
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Stop and remove container
```bash
docker stop ml-api
docker rm ml-api
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main`

### Workflow Steps

1. **Linting** (`lint` job)
   - Runs flake8 for code quality checks
   - Ensures code follows style guidelines

2. **Testing** (`test` job)
   - Runs after linting passes
   - Executes ML pipeline to generate artifacts
   - Runs pytest with coverage reporting
   - Uploads coverage to Codecov (optional)

3. **Docker Build** (`docker` job)
   - Runs after tests pass
   - Generates ML artifacts
   - Builds Docker image
   - Tests the containerized application
   - (Optional) Pushes to Docker registry

### Triggering the Full Pipeline

1. **Make code changes**
```bash
# Edit your code
git add .
git commit -m "Your commit message"
```

2. **Push to GitHub**
```bash
git push origin main
```

3. **Monitor workflow**
- Go to your repository on GitHub
- Click on the "Actions" tab
- Watch the pipeline execute automatically

### Setting Up Docker Push (Optional)

To enable Docker image push to Docker Hub:

1. Create Docker Hub account and repository

2. Add secrets to GitHub repository:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password/token

3. Uncomment the Docker login and push steps in `.github/workflows/ci-cd.yml`

## 📊 Model Performance

The Random Forest classifier achieves:
- **Cross-validation accuracy**: ~95-97%
- **Test accuracy**: ~93-97%
- **Precision/Recall/F1**: All metrics > 0.93

## 🛠️ Technologies Used

- **Python 3.10**: Core programming language
- **FastAPI**: Modern web framework for APIs
- **scikit-learn**: Machine learning library
- **pandas & numpy**: Data manipulation
- **pytest**: Testing framework
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation
- **uvicorn**: ASGI server

## 📝 Code Quality

- **Linting**: flake8 with max line length 100
- **Testing**: Comprehensive unit tests with >80% coverage
- **Type hints**: Used throughout the codebase
- **Documentation**: Detailed docstrings in all modules

## 🔮 Future Enhancements

- [ ] Add model versioning with MLflow
- [ ] Implement model retraining pipeline
- [ ] Add monitoring and logging with Prometheus/Grafana
- [ ] Support multiple ML models
- [ ] Add data drift detection
- [ ] Kubernetes deployment manifests
- [ ] API authentication and rate limiting

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a demonstration project using the Iris dataset. For production use, replace with your actual dataset and adjust the model architecture accordingly.
