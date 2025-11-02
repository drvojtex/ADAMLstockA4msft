# ADAMLstockA4msft
About Advanced Data Analysis and Machine Learning - Project Work 2

---

## 📋 **Project Description**

This project addresses the Stock A4 task within the ADAML course. The goal is to predict the closing price (Close price) of MSFT stock using multivariate time series data. 

The core focus is on implementing and comparing three advanced Deep Learning architectures suitable for time series forecasting: Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Transformer Networks. 

The required tasks include:
  - Performing comprehensive time-series analysis (decomposition, autocorrelation).
  - Designing and implementing data pretreatment suitable for deep neural networks.
  - Calibrating, validating, and comparing the RNN, LSTM, and Transformer models on multivariate data.
  - Establishing the correct time-series partitioning strategy for training and validation.

---

## 📊 **Dataset**

- The dataset contains daily trading data for Microsoft (MSFT) stock in a multivariate format.
- Data Type: Daily financial time series (Open, High, Low, Close, Volume).
- Format: CSV (internally converted to a time-indexed table).
- Structure (Columns):
  - Date (Time index)
  - Open (Daily opening price)
  - High (Daily highest price)
  - Low (Daily lowest price)
  - Close (Target variable: the price to be predicted)
  - Volume (Trading volume)
  - Name (Stock ticker: MSFT)

---
  
## 🔍 **Workflow (Project Phases Tasks)**

The project is structured into weekly phases, moving from Exploratory Data Analysis (EDA) to the implementation of the advanced Transformer model.

### 1️⃣ **Data Understanding and EDA**
- **Time-series Visualisation:** Selecting an appropriate plot for financial time-series. **(VojtaD)**
- **Time-series Decomposition:** Analysing long-term trend, seasonality, and residuals (using STL decomposition). **(Kuba)**
- **Autocorrelation Analysis (ACF/PACF):** Studying temporal dependencies in the dataset. **(VojtaV)**
- **Partitioning Plan:** Designing the strategy for dividing the time series into train, validation, and test sets. **(call together)**

### 2️⃣ **Pretreatment and Baseline Model**
- **Data Preprocessing:** Handling issues such as ensuring uniform frequency, variable synchronisation, and strategies for filling missing values.
- **Outlier Identification:** Utilising STL decomposition to spot and possibly eliminate outliers.
- **Literature Review:** Studying best practices for sub-sequencing long time series and standardising data for deep learning models (RNN/LSTM).
- **Baseline Model:** Implementation of an Autoregressive (AR) model to establish a benchmark for comparison.

### 3️⃣ **Model Planning and Architectures**
- **Detailed Plan:** Describing the implementation and mathematical aspects of the RNN, LSTM, and Transformer models.
- **Data Pretreatment Strategy:** Finalised choice and description of the pretreatment steps.
- **Model Architecture:** Definition and visualisation of the layer graph for all three deep networks.
- **Evaluation Strategy:** Metrics (e.g., RMSE, MAE, $R^2$) and analysis of residuals. Optimisation Strategy: Designing an ablation/sensitivity study for hyperparameter tuning.

### 4️⃣ **RNN and LSTM Implementation**
- **Implementation:** Calibration and training of RNN and LSTM models using the multivariate data.
- **Normalisation:** Ensuring correct data normalisation for optimal network performance.
- **Evaluation:** Presentation of training metrics and interpretation of results, explicitly addressing the complexity of multivariate data.
- **Recommendations:** Suggestions for improvement and a plan for further parameter tuning based on observed model behaviour.

### 5️⃣ **Transformer Implementation**
- **Implementation:** Calibration and training of the Multivariate Forecasting Transformer model.
- **Evaluation and Comparison:** Presentation and interpretation of the Transformer's results in comparison with RNN and LSTM.
- **Multivariate Complexity:** Discussing the model's performance on the multivariate forecasting task.
- **Final Report:** Summary and plan for future work and potential architecture enhancements.

---

## 💡 **Repository Structure**

```
ADAMLstockA4msft
│
├─ src/
│   ├─ Collection of Python helper classes for data loading, preprocessing, and analysis.
│   │   • dataloader.py: loader interface for csv files, creates pandas dataframe
│   │   • StockVisualiser.py: visualise stock time-series via plotly
│   │   • StockAutocorrelation.py: compute time-series autocorrelation
│   │   • StockDecomposition.py: time-series decomposition (trend, seasonality, residual)
│   
├─ data/
│   ├─ Raw MSFT stock data files (MSFT_2006-01-01_to_2018-01-01.csv)
│   
├─ documentation/
│   ├─ Project reports (Part1.pdf, Part2.pdf, etc. - submissions for each phase)
│
├─ main1.ipynb
│   │   • initial dataloading, analysis and visualisation
|
├─ LICENSE
│   ├─ GPL-3.0 license
│
└─ README.md
    ├─ Project overview, dataset description, workflow, and usage guide
```
