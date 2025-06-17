# VisualFlow: AI Visualization and Interpretability Tool

## Overview
VisualFlow is a comprehensive tool designed to visualize and interpret AI models. It provides interactive interfaces for exploring various machine learning models and interpretability techniques, such as SHAP and LIME. The project is built using Python and Gradio for seamless user interaction.

## Features
- **Model Visualization**:
  - K-means clustering visualization.
  - SVM decision boundary visualization.
  - CNN process visualization.
  - MLP process visualization.
  - RandomTree decision boundary visualization.
  
- **Interpretability**:
  - SHAP value estimation for feature importance.
  - LIME explanation for image classification models.

## Run the applicatio
```bash
python src/main_app.py
```
Once launched, you can interact with the tool via the Gradio interface in your browser.


## Tabs
- Visualization Tab: Select a model (e.g., K-means, SVM, CNN, MLP, RandomTree) and explore its visualization.
- Interpretability Tab: Choose an interpretability method (SHAP or LIME) to explain model predictions.

## Project Structure
```
VisualFlow/
├── src/
│   ├── main_app.py              # Entry point for the application
│   ├── interface/
│   │   ├── model/                 # Interfaces for model visualization
│   │   ├── interpret/             # Interfaces for interpretability methods
│   ├── models/                    # Model implementations (e.g., CNN, MLP, SVM)
│   ├── interpret/                 # SHAP and LIME interpreters
├── data/                          # Example data and pretrained weights
├── test/                          # Unit tests for models
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```