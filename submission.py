# submission.py

import os
import gc
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import joblib
import warnings

# --- TensorFlow and Keras Imports ---
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import backend as K


# --- Setup ---
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None


class CFG:
    N_SPLITS = 10
    N_FFT_FEATURES = 8
    ARTIFACT_DIR = Path("/kaggle/input/0-825-fix-model-cmi")


print("--- Loading models and artifacts... ---")

# --- Load Models ---
models = []
for fold in range(CFG.N_SPLITS):
    model_path = CFG.ARTIFACT_DIR / f"gesture_model_fold_{fold}.h5"
    custom_objects = {
        'time_sum': lambda x: tf.reduce_sum(x, axis=1),
        'squeeze_last_axis': lambda x: tf.squeeze(x, axis=-1),
        'expand_last_axis': lambda x: tf.expand_dims(x, axis=-1)
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    models.append(model)
print(f"Loaded {len(models)} models.")

# --- Load Scalers ---
sequence_scalers = [joblib.load(CFG.ARTIFACT_DIR / f"sequence_scaler_{fold}.pkl") for fold in range(CFG.N_SPLITS)]
static_scalers = [joblib.load(CFG.ARTIFACT_DIR / f"static_scaler_{fold}.pkl") for fold in range(CFG.N_SPLITS)]
tabular_scalers = [joblib.load(CFG.ARTIFACT_DIR / f"tabular_scaler_{fold}.pkl") for fold in range(CFG.N_SPLITS)]

# --- Load Other Artifacts ---
pad_len = np.load(CFG.ARTIFACT_DIR / "sequence_maxlen.npy").item()
gesture_classes = np.load(CFG.ARTIFACT_DIR / "gesture_classes.npy", allow_pickle=True)
final_feature_cols = np.load(CFG.ARTIFACT_DIR / "feature_cols.npy", allow_pickle=True)
static_feature_cols = np.load(CFG.ARTIFACT_DIR / "static_feature_cols.npy", allow_pickle=True)
tabular_feature_cols = np.load(CFG.ARTIFACT_DIR / "tabular_feature_cols.npy", allow_pickle=True)
imu_agg_cols = np.load(CFG.ARTIFACT_DIR / "imu_agg_cols.npy", allow_pickle=True)


print("--- Artifact loading complete. Server is ready. ---")

# ===================================================================================
# 1. FEATURE ENGINEERING HELPERS
# ===================================================================================
def remove_gravity_from_acc(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, IndexError):
            linear_accel[i, :] = acc_values[i, :]
    return linear_accel

def calculate_vertical_acceleration(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    vertical_acc = np.zeros(len(acc_values))
    gravity_world_unit = np.array([0, 0, 1.0])
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)): continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world_unit, inverse=True)
            vertical_acc[i] = np.dot(acc_values[i, :], gravity_sensor_frame)
        except (ValueError, IndexError): pass
    return vertical_acc

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((len(quat_values), 3))
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt)): continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, IndexError): pass
    return angular_vel

def calculate_angular_distance(rot_data):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(len(quat_values))
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isnan(q2)): continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except (ValueError, IndexError): pass
    return angular_dist


def get_fft_features(sequence_data, n_features):
    if len(sequence_data) < n_features * 2: return np.zeros(n_features)
    fft_vals = np.fft.fft(sequence_data)
    fft_mag = np.abs(fft_vals)[:len(fft_vals)//2]
    return fft_mag[1:n_features+1]

# ===================================================================================
# 2. FULL FEATURE ENGINEERING FUNCTION
# ===================================================================================
def engineer_features(df):
    """Applies all feature engineering steps from training."""
    df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = remove_gravity_from_acc(df, df)
    df['acc_vertical'] = calculate_vertical_acceleration(df, df)
    df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = calculate_angular_velocity_from_quat(df)
    df['angular_distance'] = calculate_angular_distance(df)
    
    quat_values = np.nan_to_num(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values, nan=[0., 0., 0., 1.])
    df[['roll', 'pitch', 'yaw']] = np.nan
    valid_quats_mask = ~np.all(quat_values == 0, axis=1)
    try:
        df.loc[valid_quats_mask, ['roll', 'pitch', 'yaw']] = R.from_quat(quat_values[valid_quats_mask]).as_euler('xyz', degrees=True)
    except (ValueError, IndexError): pass

    df['linear_acc_mag_old'] = np.linalg.norm(df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']].values, axis=1)
    df['angular_vel_mag_old'] = np.linalg.norm(df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']].values, axis=1)
    
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_sensor_data = df[pixel_cols].replace(-1, np.nan)
        df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1) 
        df[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
    
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df['linear_acc_mag'].diff().fillna(0)
    df['angular_vel_mag'] = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
    df['angular_vel_mag_jerk'] = df['angular_vel_mag'].diff().fillna(0)
    df['gesture_rhythm_signature'] = df['linear_acc_mag'].transform(lambda x: x.rolling(5, min_periods=1).std() / (x.rolling(5, min_periods=1).mean() + 1e-6))
    
    return df

# ===================================================================================
# 3. PREDICTION FUNCTION (This is called for each sequence)
# ===================================================================================
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # --- 3.1. Data Preparation ---
    sequence_df = sequence.to_pandas()
    demographics_df = demographics.to_pandas()
    df = pd.merge(sequence_df, demographics_df, on='subject', how='left')

    # --- 3.2. Validation of Raw Inputs ---
    required_raw_cols = {'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w'}
    missing_cols = required_raw_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing essential raw columns: {missing_cols}")

    # --- 3.3. Feature Engineering ---
    df_featured = engineer_features(df.copy())

    # --- 3.4. Create Model Inputs (Unscaled) ---
    # Sequence Input
    X_seq_unscaled = df_featured[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    
    # Static Input
    df_meta = df_featured.iloc[[0]]
    fft_feats = get_fft_features(df_featured['linear_acc_mag'].values, CFG.N_FFT_FEATURES)
    fft_cols = [f'fft_mag_{i+1}' for i in range(CFG.N_FFT_FEATURES)]
    for col, val in zip(fft_cols, fft_feats): df_meta[col] = val
    X_static_unscaled = df_meta[static_feature_cols].values.astype('float32')
    

    aggs = {}
    imu_agg_df = df_featured[imu_agg_cols]
    
    for col in imu_agg_cols:
        aggs[f'{col}_mean'] = imu_agg_df[col].mean()
        aggs[f'{col}_std'] = imu_agg_df[col].std()
        aggs[f'{col}_max'] = imu_agg_df[col].max()
        aggs[f'{col}_min'] = imu_agg_df[col].min()
        aggs[f'{col}_quantile'] = imu_agg_df[col].quantile(0.5) # Matches training .agg(['quantile'])
        
    # Convert dict to a single-row DataFrame and fill any potential NaNs
    aggs_df = pd.DataFrame([aggs]).fillna(0)
    
    # Select the columns in the exact order from training to create the final numpy array.
    X_tabular_unscaled = aggs_df[tabular_feature_cols].values.astype('float32')
    
    # --- 3.5. Ensemble Predictions ---
    all_preds = []
    for fold in range(CFG.N_SPLITS):
        X_seq_scaled = sequence_scalers[fold].transform(X_seq_unscaled)
        X_static_scaled = static_scalers[fold].transform(X_static_unscaled)
        X_tabular_scaled = tabular_scalers[fold].transform(X_tabular_unscaled)

        X_seq_padded = pad_sequences([X_seq_scaled], maxlen=pad_len, padding='post', dtype='float32')

        inputs = {
            'sequence_input': X_seq_padded,
            'static_input': X_static_scaled,
            'tabular_input': X_tabular_scaled,
        }
        # The model returns a dictionary; we need the 'main_output' for classification.
        pred = models[fold].predict(inputs, verbose=0)['main_output']
        all_preds.append(pred)

    # --- 3.6. Finalize Prediction ---
    avg_preds = np.mean(all_preds, axis=0)
    pred_idx = np.argmax(avg_preds)
    gesture_name = gesture_classes[pred_idx]
    
    return gesture_name