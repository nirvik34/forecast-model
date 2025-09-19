# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

CSV_PATH = '2025-grievances.csv'
SEQUENCE_LENGTH = 30
MIN_SAMPLES = 80
TF_RANDOM_SEED = 42
np.random.seed(0)
tf.random.set_seed(TF_RANDOM_SEED)

# ✅ Load only necessary columns from CSV
df = pd.read_csv(CSV_PATH, usecols=['Sub Category', 'Grievance Date'])
df['Grievance Date'] = pd.to_datetime(df['Grievance Date'])

# ✅ Hardcoded problem types
problem_types = [
    'Street Light Not Working',
    'Garbage dumping in vacant sites',
    'Potholes',
    'Garbage dump',
    'Others',
    'water stagnation',
    'Cleanliness'
]

# ✅ Keep only rows with selected problem types
df = df[df['Sub Category'].isin(problem_types)]

app = FastAPI(title="Grievance Forecast API")

origins = [
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1:5173",  # Fallback local address
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    problem: str
    date: str  # YYYY-MM-DD


@app.get("/problems")
def get_problems():
    return {"problems": problem_types}


@app.post("/predict")
def predict(request: PredictionRequest):
    selected_problem = request.problem
    sel_date = pd.to_datetime(request.date)

    if selected_problem not in problem_types:
        return {"error": f"Problem '{selected_problem}' not found"}

    # Time series preparation
    mask = df['Sub Category'] == selected_problem
    df_sub = df[mask]
    if df_sub.empty:
        return {"error": f"No data available for '{selected_problem}'"}
    daily = df_sub.groupby(df_sub["Grievance Date"].dt.date).size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['Grievance Date'])
    daily = daily[['date', 'count']]
    full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(full_range, fill_value=0).rename_axis('date').reset_index()

    # Feature engineering
    daily['month'] = daily['date'].dt.month
    daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
    daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
    daily['dayofweek'] = daily['date'].dt.dayofweek
    daily['dow_sin'] = np.sin(2 * np.pi * daily['dayofweek'] / 7)
    daily['dow_cos'] = np.cos(2 * np.pi * daily['dayofweek'] / 7)
    daily['is_festival'] = 0
    daily['is_holiday'] = 0
    daily['rain_mm'] = np.random.gamma(3, 4, size=len(daily))
    daily['count_lag1'] = daily['count'].shift(1).fillna(0)
    daily['count_lag2'] = daily['count'].shift(2).fillna(0)
    daily['log_count'] = np.log1p(daily['count'])
    daily['count_lag7'] = daily['count'].shift(7).fillna(0)
    daily['count_rolling_max7'] = daily['count'].rolling(7, min_periods=1).max().shift(1).fillna(0)
    daily['count_rolling_sum7'] = daily['count'].rolling(7, min_periods=1).sum().shift(1).fillna(0)

    features = [
        'count', 'count_lag1', 'count_lag2', 'count_lag7',
        'count_rolling_max7', 'count_rolling_sum7',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_festival', 'is_holiday', 'rain_mm'
    ]

    X_all = daily[features].values.astype(float)
    y_all = daily['log_count'].values.astype(float)

    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X_all, y_all, SEQUENCE_LENGTH)
    if len(X_seq) < MIN_SAMPLES:
        return {"error": f"Not enough data for {selected_problem}"}

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    f_scaler = MinMaxScaler()
    X_train_scaled = f_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test_scaled = f_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    t_scaler = MinMaxScaler()
    y_train_scaled = t_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    tf.random.set_seed(TF_RANDOM_SEED)
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.12),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train_scaled, y_train_scaled, epochs=60, batch_size=8, verbose=0, callbacks=[es])

    # Forecast
    start_date = daily['date'].iloc[-1]
    days_ahead = (sel_date - start_date).days
    if days_ahead <= 0:
        return {"error": "Please select a date beyond dataset range."}

    last_seq = X_all[-SEQUENCE_LENGTH:]
    last_seq_scaled = f_scaler.transform(last_seq)
    curr_seq = last_seq_scaled.copy()
    fut_preds = []

    for _ in range(days_ahead):
        pred_scaled = model.predict(curr_seq[np.newaxis, :, :], verbose=0)[0, 0]
        pred_log = t_scaler.inverse_transform([[pred_scaled]])[0, 0]
        pred_count = np.expm1(pred_log)
        fut_preds.append(pred_count)

        new_row = curr_seq[-1].copy()
        new_row[0] = pred_count
        new_row[1] = curr_seq[-1][0]
        new_row[2] = curr_seq[-1][1]
        history_counts = list(daily['count']) + fut_preds
        last7 = history_counts[-7:]
        new_row[4] = max(last7)
        new_row[5] = sum(last7)

        curr_seq = np.roll(curr_seq, -1, axis=0)
        curr_seq[-1] = new_row

    return {
        "selected_problem": selected_problem,
        "selected_date": str(sel_date.date()),
        "predicted_complaints": round(fut_preds[-1], 2)
    }
