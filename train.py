# train.py

import os, gc
import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

import joblib
import warnings

from utils import (
    remove_gravity_from_acc, calculate_vertical_acceleration, 
    calculate_angular_velocity_from_quat, calculate_angular_distance, 
    get_fft_features, IMUSpecificScaler, WarmupCosineDecay, 
    GatedMixupGenerator, EnhancedCMIMetricCallback, SWA, 
    build_monster_branch_model
)

# --- Setup ---
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None 


# --- Configuration ---
class CFG:
    TRAIN = True
    BASE_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data") 
    EXPORT_DIR = Path("./") # Output directory for models and artifacts
    SEED = 42
    BATCH_SIZE = 64
    PAD_PERCENTILE = 98
    LR_INIT = 5e-4
    WD = 3e-3
    MIXUP_ALPHA = 0.6
    EPOCHS = 180
    PATIENCE = 40
    N_SPLITS = 10
    MASKING_PROB = 0.6
    GATE_LOSS_WEIGHT = 0.2
    N_FFT_FEATURES = 8
    SWA_START_EPOCH = 110

def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if CFG.TRAIN:
    print("---------- TRAINING MODE ---------")
    # --- Load Data ---
    train_df = pd.read_csv(CFG.BASE_DIR / "train.csv")
    train_dem_df = pd.read_csv(CFG.BASE_DIR / "train_demographics.csv")
    df = pd.merge(train_df, train_dem_df, on='subject', how='left')
    print(f"Initial training data shape: {df.shape}")
    
    # --- Label Encoding ---
    le = LabelEncoder(); df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(CFG.EXPORT_DIR / "gesture_classes.npy", le.classes_)
    bfrb_gestures = [c for c in le.classes_ if ' - ' in c and not any(k in c for k in ['Wave', 'Text', 'Write', 'Feel', 'Scratch knee', 'Pull air', 'Drink', 'Glasses', 'Pinch knee'])]

    # =================================================================
    # 1. PER-TIMESTEP FEATURE ENGINEERING
    # =================================================================
    print("Performing per-timestep feature engineering...")
    df_list = []
    for _, group in tqdm(df.groupby('sequence_id'), desc="Processing Sequences"):
        group[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = remove_gravity_from_acc(group, group)
        group['acc_vertical'] = calculate_vertical_acceleration(group, group)
        group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = calculate_angular_velocity_from_quat(group)
        group['angular_distance'] = calculate_angular_distance(group)
        
        quat_values = np.nan_to_num(group[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values, nan=[0., 0., 0., 1.])
        group[['roll', 'pitch', 'yaw']] = np.nan
        valid_quats_mask = ~np.all(np.isnan(quat_values), axis=1)
        try:
            group.loc[valid_quats_mask, ['roll', 'pitch', 'yaw']] = R.from_quat(quat_values[valid_quats_mask]).as_euler('xyz', degrees=True)
        except (ValueError, IndexError):
            pass

        group['linear_acc_mag_old'] = np.linalg.norm(group[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']].values, axis=1)
        group['angular_vel_mag_old'] = np.linalg.norm(group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']].values, axis=1)
        
        for i in range(1, 6):
            pixel_cols = [f"tof_{i}_v{p}" for p in range(64)] 
            tof_sensor_data = group[pixel_cols].replace(-1, np.nan)
            group[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1) 
            group[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
        
        df_list.append(group)
    
    df = pd.concat(df_list, ignore_index=True)
    
    print("Performing per-sequence feature engineering...")
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    df['angular_vel_mag'] = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
    df['angular_vel_mag_jerk'] = df.groupby('sequence_id')['angular_vel_mag'].diff().fillna(0)
    df['gesture_rhythm_signature'] = df.groupby('sequence_id')['linear_acc_mag'].transform(lambda x: x.rolling(5, min_periods=1).std() / (x.rolling(5, min_periods=1).mean() + 1e-6))

    # =================================================================
    # 2. DEFINE AND ORDER FEATURE COLUMNS FOR THE MODEL
    # =================================================================
    print("Defining and ordering feature columns for the model...")
    
    # IMU Features (carefully ordered to match model slicing)
    acc_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
    gyro_cols = ['angular_vel_x', 'angular_vel_y', 'angular_vel_z']
    other_imu_cols = [
        'acc_vertical', 'angular_distance', 'roll', 'pitch', 'yaw', 
        'linear_acc_mag', 'angular_vel_mag', 'linear_acc_mag_jerk', 
        'angular_vel_mag_jerk', 'gesture_rhythm_signature', 'linear_acc_mag_old', 
        'angular_vel_mag_old'
    ] + sorted([c for c in df.columns if c.startswith('rot_')])
    imu_cols = acc_cols + gyro_cols + other_imu_cols
    
    # Thermopile and ToF Features
    thm_cols = sorted([c for c in df.columns if c.startswith('thm_')])
    tof_agg_cols = sorted([c for c in df.columns if c.startswith('tof_') and ('_mean' in c or '_std' in c)])
    
    final_feature_cols = imu_cols + thm_cols + tof_agg_cols
    
    imu_dim = len(imu_cols); thm_dim = len(thm_cols); tof_dim = len(tof_agg_cols)
    print(f"Sequence Dims: IMU={imu_dim} | THM={thm_dim} | TOF={tof_dim} | Total={len(final_feature_cols)}")

    # =================================================================
    # 3. CREATE STATIC AND TABULAR (AGGREGATE) FEATURES
    # =================================================================
    print("Creating static (FFT) and tabular (aggregate) features...")
    df_meta = df.groupby('sequence_id').first().reset_index()
    
    # FFT Features
    fft_features_dict = {seq_id: {f'fft_mag_{i+1}': v for i, v in enumerate(get_fft_features(seq_df['linear_acc_mag'].values, CFG.N_FFT_FEATURES))} for seq_id, seq_df in df.groupby('sequence_id')}
    fft_df = pd.DataFrame.from_dict(fft_features_dict, orient='index').reset_index().rename(columns={'index': 'sequence_id'})
    df_meta = pd.merge(df_meta, fft_df, on='sequence_id', how='left')
    
    # Tabular (Aggregate) Features - using a curated list of IMU features for robustness
    imu_agg_cols = [
        'linear_acc_mag', 'angular_vel_mag', 'angular_distance', 'acc_vertical',
        'linear_acc_mag_jerk', 'angular_vel_mag_jerk', 'gesture_rhythm_signature'
    ]
    seq_aggs = df.groupby('sequence_id')[imu_agg_cols].agg(['mean', 'std', 'max', 'min', 'quantile']).fillna(0)
    seq_aggs.columns = ['_'.join(col).strip() for col in seq_aggs.columns.values]
    df_meta = pd.merge(df_meta, seq_aggs, on='sequence_id', how='left')

    demographic_cols = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    fft_cols = [f'fft_mag_{i+1}' for i in range(CFG.N_FFT_FEATURES)]
    
    static_feature_cols = demographic_cols + fft_cols
    tabular_feature_cols = list(seq_aggs.columns)
    
    static_dim = len(static_feature_cols); tabular_dim = len(tabular_feature_cols)
    print(f"Static/Tabular Dims: Static={static_dim}, Tabular={tabular_dim}")

    # =================================================================
    # 4. ASSEMBLE FINAL MODEL INPUTS (NUMPY ARRAYS)
    # =================================================================
    print("Assembling final model inputs...")
    X_seq_raw, X_static_raw, X_tabular_raw, y_raw, groups_raw, lens_raw = [], [], [], [], [], []
    
    df_meta.set_index('sequence_id', inplace=True)
    for seq_id, seq_df in tqdm(df.groupby('sequence_id'), desc="Assembling Numpy Arrays"):
        X_seq_raw.append(seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32'))
        meta_row = df_meta.loc[seq_id]
        X_static_raw.append(meta_row[static_feature_cols].values.astype('float32'))
        X_tabular_raw.append(meta_row[tabular_feature_cols].values.astype('float32'))
        y_raw.append(meta_row['gesture_int'])
        groups_raw.append(meta_row['subject'])
        lens_raw.append(len(seq_df))

    X_seq_raw = np.array(X_seq_raw, dtype=object)
    X_static_raw = np.array(X_static_raw); X_tabular_raw = np.array(X_tabular_raw)
    y_cat_raw = to_categorical(y_raw, num_classes=len(le.classes_))
    groups_raw = np.array(groups_raw); lens_raw = np.array(lens_raw)

    pad_len = int(np.percentile(lens_raw, CFG.PAD_PERCENTILE))
    print(f"Global pad length set to: {pad_len}")

    # =================================================================
    # 5. CROSS-VALIDATION AND TRAINING LOOP
    # =================================================================
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq_raw, y_raw, groups_raw)):
        print(f"\n{'='*50}\nFOLD {fold+1}/{CFG.N_SPLITS}")
        seed_everything(CFG.SEED + fold)
        
        # --- Split Data ---
        X_train_seq_unpadded, X_val_seq_unpadded = X_seq_raw[train_idx], X_seq_raw[val_idx]
        X_train_static_unscaled, X_val_static_unscaled = X_static_raw[train_idx], X_static_raw[val_idx]
        X_train_tabular_unscaled, X_val_tabular_unscaled = X_tabular_raw[train_idx], X_tabular_raw[val_idx]
        y_train_cat, y_val_cat = y_cat_raw[train_idx], y_cat_raw[val_idx]
        y_val_gestures = le.classes_[np.array(y_raw)[val_idx]]
        
        # --- Scaling ---
        print("  Fitting fold-specific scalers...")
        sequence_scaler = IMUSpecificScaler().fit(np.concatenate(X_train_seq_unpadded, axis=0), imu_dim)
        static_scaler = StandardScaler().fit(X_train_static_unscaled)
        tabular_scaler = StandardScaler().fit(X_train_tabular_unscaled)
        
        X_train_seq_scaled = [sequence_scaler.transform(x) for x in X_train_seq_unpadded]
        X_val_seq_scaled = [sequence_scaler.transform(x) for x in X_val_seq_unpadded]
        X_train_static = static_scaler.transform(X_train_static_unscaled); X_val_static = static_scaler.transform(X_val_static_unscaled)
        X_train_tabular = tabular_scaler.transform(X_train_tabular_unscaled); X_val_tabular = tabular_scaler.transform(X_val_tabular_unscaled)
        
        # --- Padding ---
        X_train = pad_sequences(X_train_seq_scaled, maxlen=pad_len, padding='post', dtype='float32')
        X_val = pad_sequences(X_val_seq_scaled, maxlen=pad_len, padding='post', dtype='float32')
        
        # --- Model Build and Compile ---
        K.clear_session()
        model = build_monster_branch_model(
            pad_len=pad_len, 
            imu_dim=imu_dim, 
            thm_dim=thm_dim, 
            tof_dim=tof_dim, 
            static_dim=static_dim, 
            tabular_dim=tabular_dim, 
            n_classes=len(le.classes_), 
            wd=CFG.WD
        )
        
        lr_schedule = WarmupCosineDecay(CFG.LR_INIT, (len(X_train) // CFG.BATCH_SIZE) * CFG.EPOCHS, int((len(X_train) // CFG.BATCH_SIZE) * CFG.EPOCHS * 0.1))
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=CFG.WD)
        model.compile(
            optimizer=optimizer,
            loss={'main_output': CategoricalCrossentropy(label_smoothing=0.1), 'tof_gate': 'binary_crossentropy'},
            loss_weights={'main_output': 1.0, 'tof_gate': CFG.GATE_LOSS_WEIGHT}
        )
        
        # --- Data Generator ---
        # NOTE: You MUST update your GatedMixupGenerator to accept a 3-element tuple for X
        class_weights = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_train_cat.argmax(1))
        train_gen = GatedMixupGenerator(
            (X_train, X_train_static, X_train_tabular), # Tuple of 3 inputs
            y_train_cat, CFG.BATCH_SIZE, imu_dim, dict(enumerate(class_weights)),
            alpha=CFG.MIXUP_ALPHA, masking_prob=CFG.MASKING_PROB
        )
        
        val_inputs = {
            'sequence_input': X_val, 
            'static_input': X_val_static, 
            'tabular_input': X_val_tabular
        }

        # --- Callbacks and Training ---
        callbacks = [
            EnhancedCMIMetricCallback(val_inputs, y_val_gestures, le.classes_, bfrb_gestures, imu_dim, patience=CFG.PATIENCE),
            SWA(start_epoch=CFG.SWA_START_EPOCH)
        ]
        
        print(f"  Starting training for {CFG.EPOCHS} epochs...")
        model.fit(
            train_gen, 
            epochs=CFG.EPOCHS, 
            validation_data=(val_inputs, {'main_output': y_val_cat, 'tof_gate': np.ones(len(y_val_cat))}), 
            callbacks=callbacks, 
            verbose=0
        ) 

        # --- Save Artifacts ---
        if fold == 0:
            print("\n  Saving artifacts from Fold 1 for inference...")
            np.save(CFG.EXPORT_DIR / "sequence_maxlen.npy", np.array([pad_len]))
            np.save(CFG.EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))
            np.save(CFG.EXPORT_DIR / "static_feature_cols.npy", np.array(static_feature_cols))
            np.save(CFG.EXPORT_DIR / "tabular_feature_cols.npy", np.array(tabular_feature_cols))
            np.save(CFG.EXPORT_DIR / "imu_agg_cols.npy", np.array(imu_agg_cols)) # Save the agg cols list

        model.save(CFG.EXPORT_DIR / f"gesture_model_fold_{fold}.h5")
        joblib.dump(sequence_scaler, CFG.EXPORT_DIR / f"sequence_scaler_{fold}.pkl")
        joblib.dump(static_scaler, CFG.EXPORT_DIR / f"static_scaler_{fold}.pkl")
        joblib.dump(tabular_scaler, CFG.EXPORT_DIR / f"tabular_scaler_{fold}.pkl")

        print(f"\nFOLD {fold+1} COMPLETED ✓ Model and artifacts saved.")

        del model, X_train, X_val, train_gen; gc.collect()
        K.clear_session()

    print("\n---- TRAINING COMPLETE ----")