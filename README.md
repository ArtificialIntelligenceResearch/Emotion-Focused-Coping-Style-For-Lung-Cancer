
<div align="center">
      <h1><br/>Emotion Focused Coping Style For Lung Cancer</h1>
     </div>

<body>
<p align="center">
  <a href="mailto:ahammadmejbah@gmail.com"><img src="https://img.shields.io/badge/Email-ahammadmejbah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/BytesOfIntelligences"><img src="https://img.shields.io/badge/GitHub-%40BytesOfIntelligences-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://linkedin.com/in/ahammadmejbah"><img src="https://img.shields.io/badge/LinkedIn-Mejbah%20Ahammad-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://bytesofintelligences.com/"><img src="https://img.shields.io/badge/Website-Bytes%20of%20Intelligence-lightgrey?style=flat-square&logo=google-chrome"></a>
  <a href="https://www.youtube.com/@BytesOfIntelligences"><img src="https://img.shields.io/badge/YouTube-BytesofIntelligence-red?style=flat-square&logo=youtube"></a>
  <a href="https://www.researchgate.net/profile/Mejbah-Ahammad-2"><img src="https://img.shields.io/badge/ResearchGate-Mejbah%20Ahammad-blue?style=flat-square&logo=researchgate"></a>
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801874603631-green?style=flat-square&logo=whatsapp">
  <a href="https://www.hackerrank.com/profile/ahammadmejbah"><img src="https://img.shields.io/badge/Hackerrank-ahammadmejbah-green?style=flat-square&logo=hackerrank"></a>
</p>


### This repository contains the code and data for analyzing emotion-focused coping styles in patients with lung cancer.

This study examined the relationships between coping styles, stress levels, and pain levels in lung cancer patients, and whether these relationships were moderated by place of residence (small town vs large city). A sample of 97 lung cancer patients completed questionnaires measuring coping styles (Stress Coping Inventory), perceived stress levels (Perceived Stress Scale), and pain levels (Short Form McGill Pain Questionnaire).

Results showed that emotion-focused coping style was positively associated with higher stress and pain levels, while task-focused coping style was negatively associated with stress levels. Stress levels fully mediated the relationship between emotion-focused coping and pain. Place of residence moderated these relationships - the associations between emotion-focused coping, stress, and pain were stronger for patients residing in small towns compared to those in larger cities.

The findings suggest that an emotion-focused way of coping raises stress levels in lung cancer patients, which in turn increases experiences of pain. This pattern is especially pronounced among patients from rural areas and small towns. In contrast, a task-oriented coping approach is linked to lower stress. Stress acts as a key mechanism explaining how different coping strategies influence pain perception in lung cancer.

The results highlight the importance of psychological interventions aimed at improving coping skills, particularly for rural patients who may have less access to medical resources and support. Shifting from an emotion-focused to a task-oriented coping style could help reduce stress and mitigate pain among lung cancer patients. Further longitudinal research is needed to better understand the impacts of residential location on patient experiences over the course of treatment.
### Repository Structure
```
Emotion-Focused-Coping-Style-For-Lung-Cancer/
│
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── utils.py
│   ├── README.md
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   ├── README.md
├── docs/
│   ├── overview.md
│   ├── data_description.md
│   ├── model_architecture.md
│   ├── evaluation_metrics.md
│   ├── setup_guide.md
│   ├── usage_guide.md
│   ├── contributing.md
│   ├── LICENSE.md
│   ├── acknowledgments.md
│   ├── troubleshooting.md
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE
```




## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts and Notebooks](#scripts-and-notebooks)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![GitHub issues](https://img.shields.io/github/issues/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![GitHub](https://img.shields.io/github/license/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer)
![Python](https://img.shields.io/badge/python-v3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-orange)
![Keras](https://img.shields.io/badge/Keras-2.8-red)
![PyTorch](https://img.shields.io/badge/PyTorch-1.11-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24-blue)
![NumPy](https://img.shields.io/badge/numpy-1.21-blue)
## Project Overview

Lung cancer patients often experience significant emotional distress. This project aims to analyze and understand the emotion-focused coping styles adopted by these patients using machine learning techniques. The ultimate goal is to provide insights that can improve psychological interventions and support.

For a detailed project overview, please refer to the [project overview documentation](docs/overview.md).

## Data

The data used in this project is sourced from clinical studies involving lung cancer patients. It includes survey responses, clinical records, and other relevant information. 

### Data Structure

- `data/raw`: Contains raw data files as received from the data sources.
- `data/processed`: Contains cleaned and processed data ready for analysis.

For more details on the data, refer to the [data description documentation](docs/data_description.md).

## Installation

To get started with this project, follow these steps:

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Emotion-Focused-Coping-Style-For-Lung-Cancer.git
cd Emotion-Focused-Coping-Style-For-Lung-Cancer
```

### Step 2: Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies. You can create one using `venv`:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### Step 4: Install Additional Libraries

If specific libraries like TensorFlow, PyTorch, or Keras need to be installed, follow the instructions below:

#### TensorFlow

```bash
pip install tensorflow==2.8
```

#### Keras

```bash
pip install keras==2.8
```

#### PyTorch

```bash
pip install torch==1.11 torchvision==0.12
```

### Step 5: Verify the Installation

Ensure all necessary libraries are installed correctly by running:

```bash
python -c "import tensorflow as tf; import keras; import torch; import sklearn; import numpy"
```

For detailed installation instructions, refer to the [setup guide documentation](docs/setup_guide.md).

## Usage

### Data Preprocessing

To preprocess the raw data, run:

```bash
python src/data_preprocessing.py
```

### Model Training

To train the models, run:

```bash
python src/model_training.py
```

### Evaluation

To evaluate the models, run:

```bash
python src/evaluation.py
```

For detailed usage instructions, refer to the [usage guide documentation](docs/usage_guide.md).

## Scripts and Notebooks

- `src/data_preprocessing.py`: Script for preprocessing raw data.
- `src/model_training.py`: Script for training machine learning models.
- `src/evaluation.py`: Script for evaluating model performance.
- `notebooks/data_exploration.ipynb`: Jupyter notebook for initial data exploration and visualization.
- `notebooks/model_analysis.ipynb`: Jupyter notebook for detailed model analysis.

For more details on each script and notebook, refer to their respective README files in the `src` and `notebooks` directories.

## Documentation

- [Project Overview](docs/overview.md)
- [Data Description](docs/data_description.md)
- [Model Architecture](docs/model_architecture.md)
- [Evaluation Metrics](docs/evaluation_metrics.md)
- [Setup Guide](docs/setup_guide.md)
- [Usage Guide](docs/usage_guide.md)
- [Contributing](docs/contributing.md)
- [License](docs/LICENSE.md)
- [Acknowledgments](docs/acknowledgments.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contributing

Contributions are welcome! Please read the [contributing guide documentation](docs/contributing.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the clinical researchers and patients who contributed to the data collection for this study.

For more details, refer to the [acknowledgments documentation](docs/acknowledgments.md).


### Model Architecture

#### docs/model_architecture.md

# Model Architecture

This document provides a detailed description of the machine learning models used in the Emotion-Focused Coping Style For Lung Cancer project.

## Overview

We employ several machine learning models to analyze the emotion-focused coping styles of lung cancer patients. The models include:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Gradient Boosting
5. Neural Networks

## Logistic Regression

Logistic Regression is a linear model used for binary classification problems. It calculates the probability that a given input point belongs to a certain class.

### Implementation

The logistic regression model is implemented using `scikit-learn`'s `LogisticRegression` class.

```python
from sklearn.linear_model import LogisticRegression

# Initialize the model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
predictions = log_reg.predict(X_test)
```

## Support Vector Machine (SVM)

SVM is a powerful classification algorithm that works by finding the hyperplane that best separates the classes in the feature space.

### Implementation

The SVM model is implemented using `scikit-learn`'s `SVC` class.

```python
from sklearn.svm import SVC

# Initialize the model
svm = SVC()

# Train the model
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)
```

## Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees and merges them together to get a more accurate and stable prediction.

### Implementation

The Random Forest model is implemented using `scikit-learn`'s `RandomForestClassifier` class.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y

_train)

# Make predictions
predictions = rf.predict(X_test)
```

## Gradient Boosting

Gradient Boosting is another ensemble technique that builds models sequentially, with each new model attempting to correct the errors made by the previous models.

### Implementation

The Gradient Boosting model is implemented using `scikit-learn`'s `GradientBoostingClassifier` class.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Initialize the model
gb = GradientBoostingClassifier()

# Train the model
gb.fit(X_train, y_train)

# Make predictions
predictions = gb.predict(X_test)
```

## Neural Networks

Neural Networks are a set of algorithms, modeled loosely after the human brain, designed to recognize patterns.

### Implementation

The Neural Network model is implemented using `TensorFlow`'s `Keras` API.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X_test)
```

## Conclusion

Each of these models provides unique advantages and can be used based on the specific requirements of the analysis. The choice of model depends on factors such as the nature of the data, the importance of interpretability, and computational resources available.

For detailed explanations and theoretical backgrounds, refer to the following resources:
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Neural Networks](https://www.tensorflow.org/guide/keras/sequential_model)

