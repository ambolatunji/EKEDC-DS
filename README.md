
# **EKEDC Smart Meter Prediction System**  
A fully-integrated **smart meter prediction and forecasting** tool for revenue loss, anomaly detection, energy consumption, and other key metrics using machine learning and deep learning models. This app supports **single prediction**, **batch prediction**, **real-time forecasting**, and **model evaluation**.

## **Features**  
- **Single Prediction**: Predict anomaly status and forecast revenue loss or recovery based on user inputs.
- **Batch Prediction**: Upload a CSV file with base inputs and predict multiple meters at once.
- **Dynamic Feature Engineering**: Automatically derive necessary features like energy loss, efficiency ratio, and others from user inputs.
- **Forecasting**: Predict revenue loss or recovery scenarios over 48 hours, including uncertainty bands.
- **Risk & Confidence Scoring**: Provide detailed risk scores and confidence scores for predictions.
- **Model Evaluation**: Evaluate model performance with metrics like accuracy, precision, recall, ROC-AUC, confusion matrices, and more.
- **Visualization**: Includes line graphs, bar charts, pie charts, and map-based heatmaps for performance analysis.
- **Admin Panel**: Track predictions, logs, and audit history for predictions and model training.
  
---

## **Installation**  
Follow these steps to get the app running locally.

### Prerequisites  
- **Python 3.8+**
- **Libraries**:  
  You can install the necessary libraries via pip:
  
```bash
pip install -r requirements.txt
```

### File Requirements  
1. **Model Files** (`models/`):  
   Upload or generate your trained machine learning and deep learning models in the **models** directory. Models should be saved as `.pkl` for traditional machine learning models or `.h5` for deep learning models.

2. **Feature File** (`_features.txt`):  
   A **.txt** file containing the list of features used in the model, one per line. This file is required for training and prediction.

3. **Data Files**:  
   You should have your **daily_meter_summary.csv** file for feature extraction, or you can upload a new dataset for predictions.

---

## **How to Use the App**

1. **Run the Streamlit App**  
   After installing dependencies, run the app using:
   
```bash
streamlit run app.py
```

2. **Single Prediction**  
   - Select the **Band** (A, B, C, D, E) for tariff selection.  
   - Enter the **Total Appliance Watt** and **Usage Hours** for your appliance.  
   - Optionally, input **Power Factor** (default is 0.95).  
   - The system will auto-calculate derived features such as **energy loss**, **revenue loss**, **energy efficiency ratio**, etc.  
   - Press the **Predict** button to see the **Risk Score**, **Confidence Score**, and **48-hour Forecast** for **Revenue Loss / Recovery**.

3. **Batch Prediction**  
   - Upload a **CSV file** containing **meter_id**, **total_watt**, **usage_hours**, **band**, and **optional power_factor** columns.  
   - The app will predict the **risk scores** and **revenue losses** for all meters in the uploaded batch.
   - Results will be available for download as **batch_predictions_DATE.csv**.

4. **Forecasting**  
   - **Revenue Loss / Recovery** scenario forecasting can be visualized for **48 hours** in the future based on your inputs or predicted anomalies.

5. **Evaluation & Logs**  
   - View the **model evaluation metrics** (e.g., ROC-AUC, confusion matrix) and predictions.
   - Access the **audit logs** to track the prediction and training history in the **Admin Panel**.

---

## **File Structure**
Here’s a quick look at the file structure:

```
.
├── app.py                # Main Streamlit application
├── train_model.py        # Script for training models
├── train_models_app.py   # Streamlit app for training models
├── evaluate_model.py     # Script for evaluating model performance
├── evaluate_model_app.py # Streamlit app for evaluating models
├── predict_app.py        # App for making predictions (single + batch)
├── prepare_feature.py    # Feature engineering logic
├── feature_engineering.py# Script to handle feature calculation
├── requirements.txt      # Required Python packages
└── models/               # Trained machine learning models
```

---

## **Key Functions**

### **Prediction:**
- **Single Prediction**:  
  The user inputs basic values like **Watt**, **Usage Hours**, and **Band** selection. The app will predict the anomaly and give a forecast of revenue loss or recovery for 48 hours.

- **Batch Prediction**:  
  The app supports uploading a **CSV file** containing meters' base inputs, and it will predict for all meters. Results are saved to a CSV file that can be downloaded.

### **Model Evaluation**:  
- Model evaluation includes several key metrics like **Confusion Matrix**, **ROC Curve**, and **AUC Score** to evaluate prediction accuracy.

### **Forecasting**:  
- **Revenue Loss / Recovery** forecasts over the next 48 hours based on user inputs or predicted anomalies.

### **Admin Panel & Logs**:  
- **Logs and Audit History** track model training sessions, prediction requests, and errors.

---

## **Important Notes**
- The model will **auto-calculate** derived features during prediction (e.g., `energy_loss_kwh_sum`, `energy_efficiency_ratio`, etc.) based on user inputs.
- **Forecast scenarios** (e.g., revenue loss and recovery) will be automatically displayed after prediction.
- **Risk Scores and Confidence Scores** will always be displayed as percentages and capped between 0-100%.
- If **band is selected** (A, B, C, D, or E), the **tariff value** will be automatically assigned and used for prediction/forecasting.

---

## **Sample CSV Template for Batch Prediction**

```csv
meter_id,total_watt,usage_hours,band,power_factor
EKEDC_000001,1000,24,A,0.95
EKEDC_000002,1500,20,B,0.95
...
```

You can download this template and fill in your data for batch prediction.

---

## **Future Enhancements**

- **Real-time Prediction Updates**: Enabling Slack or Email notifications for alerts on high-risk predictions.
- **Optimization**: Work on faster batch prediction performance.
- **Integration**: Connect to external data sources for automated updates of smart meter data.

---

## **Contributing**

Feel free to fork this repository and contribute any improvements. Pull requests are always welcome!

---

### **License**
This project is licensed under the MIT License.
