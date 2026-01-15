# Insurance Charges Prediction

A machine learning project that predicts medical insurance charges based on personal and health-related features using Linear Regression. This repository includes a Jupyter Notebook for data analysis and model training, as well as a Streamlit web application for interactive predictions.

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the insurance dataset including visualizations and statistical summaries.
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature engineering, and scaling.
- **Feature Selection**: Correlation analysis and chi-square tests to identify significant features.
- **Model Training**: Linear Regression model with evaluation metrics (R² and Adjusted R²).
- **Interactive Web App**: Streamlit application for real-time insurance charge predictions.
- **Model Persistence**: Save and load trained models for deployment.

## 📋 Requirements

- Python 3.7+
- Libraries: numpy, pandas, seaborn, matplotlib, scikit-learn, scipy, joblib, streamlit

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/deepseek23/insurance-prediction.git
   cd insurance-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `insurance.csv` dataset in the project directory or update the path in the notebook accordingly.

## 📊 Dataset

The project uses the [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle. It contains information about individuals' medical insurance charges based on features like age, BMI, number of children, smoking status, gender, and region.

**Features:**
- age: Age of the individual
- sex: Gender (male/female)
- bmi: Body Mass Index
- children: Number of children
- smoker: Smoking status (yes/no)
- region: Residential region
- charges: Medical insurance charges (target variable)

## 🔬 Usage

### Running the Jupyter Notebook

1. Open the `ml_1.ipynb` notebook in Jupyter Lab or Jupyter Notebook.
2. Execute the cells sequentially to perform EDA, data preprocessing, feature engineering, model training, and evaluation.
3. The notebook will generate and save the trained model files (`linear_regression_model.pkl`, `scaler.pkl`, `columns.pkl`) required for the Streamlit app.

### Running the Streamlit App

After running the notebook to generate the model files:

1. Ensure you have the saved model files in the project directory.
2. Run the Streamlit application:
   ```bash
   streamlit run insurance_app.py
   ```
3. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).
4. Enter your personal information in the input fields and click "Predict Charges" to get an estimate of insurance costs.

## 🤖 Model Details

- **Algorithm**: Linear Regression
- **Preprocessing**: Standard scaling for numerical features (age, BMI, children)
- **Feature Engineering**: BMI categorization, one-hot encoding for categorical variables
- **Evaluation Metrics**:
  - R² Score: Measures the proportion of variance explained by the model
  - Adjusted R²: Adjusted for the number of predictors in the model

## 📈 Results

The trained model achieves an R² score of approximately 0.78 on the test set, indicating a good fit for predicting insurance charges.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

If you have any questions or suggestions, please open an issue on GitHub.