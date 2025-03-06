# Neural Network-Based Stock Market Index Forecasting

This repository contains my Python implementation of neural network models for forecasting the S&P 500 index. I developed this project to explore and compare the performance of five neural network architectures—LSTM, CNN, ANN, RNN, and GRU—using historical S&P 500 data to predict future closing prices. The evaluation is based on RMSE, MAPE, and Directional Accuracy metrics.

## Project Overview

This project forecasts the S&P 500 index using daily closing prices from January 1, 2018, to December 30, 2023. I designed and trained five neural network models, implementing data retrieval, preprocessing, model training, and evaluation from scratch. The goal is to assess how well these models predict stock market trends and provide insights into their predictive capabilities.

### Features
- **Data**: S&P 500 closing prices sourced from Yahoo Finance.
- **Models**: LSTM, CNN, ANN, RNN, and GRU with custom architectures.
- **Metrics**: Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), and Directional Accuracy (DA).
- **Visualization**: Plot of actual vs. predicted test data for all models.

## Prerequisites

To run this project, ensure you have the following installed:
- **Python**: 3.7 or higher (tested with 3.11.7).
- **Libraries**:
  - `numpy`
  - `pandas`
  - `yfinance`
  - `tensorflow` (2.3.1 or compatible version)
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

You can install the dependencies using pip:
```bash
pip install numpy pandas yfinance tensorflow scikit-learn matplotlib seaborn
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/siddhunayak/stock_forecasting.git
   cd stock_forecasting
   ```

2. **Install Dependencies**:
   Run the command above to install required Python packages.

3. **Verify Setup**:
   Ensure you have an active internet connection (for data download via yfinance) and sufficient computational resources (CPU/GPU) for training neural networks.

## Usage

1. **Run the Code**:
   The main script is provided in a Jupyter Notebook (`stock_forecasting.ipynb`). Open it in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook stock_forecasting.ipynb
   ```
   Alternatively, convert it to a Python script (`stock_forecasting.py`) and run:
   ```bash
   python stock_forecasting.py
   ```

2. **What to Expect**:
   - The script downloads S&P 500 data from January 1, 2018, to December 30, 2023.
   - It trains five neural network models (LSTM, CNN, ANN, RNN, GRU) with a 68/32 train-test split.
   - Performance metrics (RMSE, MAPE, DA) are printed for training and testing phases.
   - A plot comparing actual test data with predictions from all models is displayed.

3. **Customization**:
   - Modify the date range in `download_data()` to analyze different periods.
   - Adjust model architectures or hyperparameters in the respective `build_*()` functions.
   - Change the train-test split ratio in `prepare_data()`.

## Project Structure

```
├── stock_forecasting.ipynb  # Main Jupyter Notebook with implementation
├── README.md               # Project documentation (this file)

```

## Methodology

### Data
- **Source**: Yahoo Finance (`^GSPC` ticker).
- **Period**: January 1, 2018, to December 30, 2023 (1509 data points).
- **Preprocessing**: Normalized with MinMaxScaler; sliding window of 3 days.

### Models
- **LSTM**: 50 units, tanh activation, Dense(1).
- **CNN**: Conv1D (64 filters, kernel size 2, ReLU), MaxPooling1D (pool size 2), Flatten, Dense(1).
- **ANN**: Dense (50 units, ReLU), Flatten, Dense(1).
- **RNN**: SimpleRNN (50 units, tanh), Dense(1).
- **GRU**: GRU (50 units, tanh), Dense(1).
- All models use the Adam optimizer and MSE loss.

### Training
- **Epochs**: 500 (with EarlyStopping, patience=50).
- **Batch Size**: 128.
- **Split**: 68% training, 32% testing.

### Evaluation
- **RMSE**: Measures prediction error magnitude.
- **MAPE**: Percentage error with epsilon (1e-10) for stability.
- **Directional Accuracy**: Percentage of correct trend predictions.

## Results

The code outputs performance metrics for each model based on my implementation. Example results (unscaled):
- **RMSE (Test)**: RNN (~53.80), ANN (~54.59), GRU (~58.84), LSTM (~61.93), CNN (~62.98).
- **MAPE (Test)**: RNN (~1.04%), ANN (~1.05%), GRU (~1.13%), LSTM (~1.18%), CNN (~1.21%).
- **DA (Test)**: CNN (~49.69%), GRU (~47.61%), RNN (~48.23%), LSTM (~48.23%), ANN (~46.78%).



