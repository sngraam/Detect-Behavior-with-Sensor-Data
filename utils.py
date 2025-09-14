# untils.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R


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
            # BUG FIX: Changed from a single faulty line to two separate assignments.
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


# Metric & Evaluation
def calculate_competition_metric(y_true_gestures, y_pred_gestures, bfrb_gestures):
    y_true_bin = np.isin(y_true_gestures, bfrb_gestures).astype(int)
    y_pred_bin = np.isin(y_pred_gestures, bfrb_gestures).astype(int)
    f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0, average='binary')
    
    def map_to_mc_labels(gestures, bfrb_gestures):
        return np.array([g if g in bfrb_gestures else 'non_target' for g in gestures])
    
    y_true_mc = map_to_mc_labels(y_true_gestures, bfrb_gestures)
    y_pred_mc = map_to_mc_labels(y_pred_gestures, bfrb_gestures)
    f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
    return 0.5 * f1_binary + 0.5 * f1_macro, f1_macro


def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed

class IMUSpecificScaler:
    def __init__(self): self.imu_scaler = StandardScaler(); self.tof_scaler = StandardScaler(); self.imu_dim = None
    def fit(self, X, imu_dim):
        self.imu_dim = imu_dim; tof_thm_data = X[:, imu_dim:]
        valid_tof_thm_mask = np.any(tof_thm_data != 0, axis=1)
        if np.sum(valid_tof_thm_mask) > 0: self.tof_scaler.fit(tof_thm_data[valid_tof_thm_mask])
        self.imu_scaler.fit(X[:, :imu_dim]); return self
    def transform(self, X):
        X_imu = self.imu_scaler.transform(X[:, :self.imu_dim]); X_tof_thm = X[:, self.imu_dim:].copy()
        transform_mask = np.any(X_tof_thm != 0, axis=1)
        if np.sum(transform_mask) > 0: X_tof_thm[transform_mask] = self.tof_scaler.transform(X_tof_thm[transform_mask])
        return np.concatenate([X_imu, X_tof_thm], axis=1)

def evaluate_dual_cmi_metric(model, val_inputs, y_val_gestures, gesture_classes, bfrb_gestures, imu_dim):
    """
    Evaluates the model on both full-sensor and IMU-only data.
    
    Args:
        model: The trained Keras model.
        val_inputs (dict): A dictionary of validation numpy arrays matching the model's input names.
        y_val_gestures (np.array): The ground truth gesture strings for the validation set.
        ... (other parameters)
    """
    # --- Full Sensor Score ---
    # The val_inputs dictionary is passed directly to the predict function
    full_preds_idx = model.predict(val_inputs, verbose=0, batch_size=256)['main_output'].argmax(axis=1)
    full_gestures = gesture_classes[full_preds_idx]
    full_sensor_score, full_marco = calculate_competition_metric(y_val_gestures, full_gestures, bfrb_gestures)

    # --- IMU-Only Score ---
    # Create a copy of the inputs to modify for the IMU-only prediction
    imu_inputs = {key: arr.copy() for key, arr in val_inputs.items()}
    # Zero out the ToF/Thermopile features in the sequence input
    imu_inputs['sequence_input'][:, :, imu_dim:] = 0.0
    
    imu_preds_idx = model.predict(imu_inputs, verbose=0, batch_size=256)['main_output'].argmax(axis=1)
    imu_gestures = gesture_classes[imu_preds_idx]
    imu_only_score, imu_marco = calculate_competition_metric(y_val_gestures, imu_gestures, bfrb_gestures)
    
    return {
        'full_sensor_score': full_sensor_score,
        'imu_only_score': imu_only_score,
        'composite_score': (full_sensor_score + imu_only_score) / 2.0,
        'full_macro_f1': full_marco,
        'imu_macro_f1': imu_marco
    }


class EnhancedCMIMetricCallback(tf.keras.callbacks.Callback):
    """
    A Keras callback to evaluate the custom competition metric at the end of each epoch.
    It now robustly handles multi-input models by accepting a dictionary of validation data.
    """
    def __init__(self, val_inputs, y_val_gestures, gesture_classes, bfrb_gestures, imu_dim, patience=40, verbose=1):
        super().__init__()
        # <-- Accepts a single dictionary of all validation inputs
        self.val_inputs = val_inputs 
        self.y_val_gestures = y_val_gestures
        self.gesture_classes = gesture_classes
        self.bfrb_gestures = bfrb_gestures
        self.imu_dim = imu_dim
        self.patience = patience
        self.verbose = verbose
        self.best_composite_score = -np.inf
        self.best_full_sensor_score = -np.inf
        self.best_imu_only_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # <-- Pass the entire validation inputs dictionary to the evaluation function
        dual_scores = evaluate_dual_cmi_metric(
            self.model, 
            self.val_inputs, 
            self.y_val_gestures, 
            self.gesture_classes, 
            self.bfrb_gestures, 
            self.imu_dim
        )
        
        logs = logs or {}
        logs.update({f'val_{k}': v for k, v in dual_scores.items()})
        
        if self.verbose > 0:
            print(f" - val_full: {dual_scores['full_sensor_score']:.4f} - val_imu: {dual_scores['imu_only_score']:.4f} - val_composite: {dual_scores['composite_score']:.4f} - val_full_macro: {dual_scores['full_macro_f1']:.4f} - val_imu_macro: {dual_scores['imu_macro_f1']:.4f}", end='')
            
        if dual_scores['composite_score'] > self.best_composite_score:
            self.best_composite_score = dual_scores['composite_score']
            self.best_full_sensor_score = dual_scores['full_sensor_score']
            self.best_imu_only_score = dual_scores['imu_only_score']
            self.wait = 0
            self.best_weights = self.model.get_weights()
            if self.verbose > 0:
                print(' *', end='', flush=True)
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.model.stop_training = True
            print(f'\nEpoch {epoch + 1:05d}: early stopping')

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.verbose > 0:
            print(f"\nTraining ended. Best Composite Score: {self.best_composite_score:.4f} (Full: {self.best_full_sensor_score:.4f}, IMU: {self.best_imu_only_score:.4f})")

import numpy as np
import tensorflow as tf

class GatedMixupGenerator(tf.keras.utils.Sequence):
    """
    A sophisticated data generator for Keras that handles:
    1. Multi-input models (sequence, static, tabular, subject).
    2. Gating mechanism for sensor data.
    3. Sensor Dropout for regularization.
    4. Mixup augmentation for continuous features.
    5. Class weighting for imbalanced datasets.
    """
    def __init__(self, X, y, batch_size, imu_dim, class_weight=None, alpha=0.8, masking_prob=0.6, sensor_dropout_rate=0.02):
        """
        Initializes the generator.
        
        Args:
            X (tuple): A tuple of numpy arrays (X_seq, X_static, X_tabular, X_subject).
            y (np.array): The one-hot encoded labels.
            batch_size (int): The size of each batch.
            imu_dim (int): The number of IMU features in the sequence data.
            class_weight (dict): A dictionary mapping class indices to their weights.
            alpha (float): The alpha parameter for the Beta distribution in Mixup.
            masking_prob (float): The probability of masking ToF/Thm sensors to train the gate.
            sensor_dropout_rate (float): The probability of dropping full-sensor data as regularization.
        """
        # Unpack the tuple of input arrays
        self.X_seq, self.X_static, self.X_tabular = X
        self.y = y
        
        # Store configuration
        self.batch_size = batch_size
        self.imu_dim = imu_dim
        self.class_weight = class_weight
        self.alpha = alpha
        self.masking_prob = masking_prob
        self.sensor_dropout_rate = sensor_dropout_rate
        
        # Initialize indices for shuffling
        self.indices = np.arange(len(self.X_seq))
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.X_seq) / self.batch_size))

    def __getitem__(self, batch_index):
        """Generates one batch of data."""
        # Get indices for the current batch
        start_idx = batch_index * self.batch_size
        end_idx = (batch_index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Retrieve the batch data for all inputs and labels
        Xb_seq = self.X_seq[batch_indices].copy()
        Xb_static = self.X_static[batch_indices].copy()
        Xb_tabular = self.X_tabular[batch_indices].copy()
        yb = self.y[batch_indices].copy()
        
        # Initialize sample weights and gate targets
        batch_size = len(Xb_seq)
        sample_weights = np.ones(batch_size, dtype='float32')
        if self.class_weight:
            sample_weights = np.array([self.class_weight[i] for i in yb.argmax(axis=1)])
        
        gate_target = np.ones(batch_size, dtype='float32')
        
        # --- Stage 1: Augmentations (Masking & Sensor Dropout) ---
        for i in range(batch_size):
            # Apply `masking_prob` to train the gate on how to handle missing data
            if np.random.rand() < self.masking_prob:
                Xb_seq[i, :, self.imu_dim:] = 0
                gate_target[i] = 0.0
            
            # Apply `sensor_dropout` as a regularization technique
            elif gate_target[i] == 1.0 and np.random.rand() < self.sensor_dropout_rate:
                Xb_seq[i, :, self.imu_dim:] = 0
                gate_target[i] = 0.0 # Gate target must also be zeroed

        # --- Stage 2: Mixup ---
        if self.alpha > 0:
            # Get mixing coefficient and permutation indices
            lam = np.random.beta(self.alpha, self.alpha)
            perm_indices = np.random.permutation(batch_size)
            
            # Apply mixup to all continuous data types
            X_seq_mix = lam * Xb_seq + (1 - lam) * Xb_seq[perm_indices]
            X_static_mix = lam * Xb_static + (1 - lam) * Xb_static[perm_indices]
            X_tabular_mix = lam * Xb_tabular + (1 - lam) * Xb_tabular[perm_indices]
            y_mix = lam * yb + (1 - lam) * yb[perm_indices]
            gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm_indices]
            sample_weights_mix = lam * sample_weights + (1 - lam) * sample_weights[perm_indices]
            
            # ** CRITICAL: Do NOT mix up the discrete subject IDs. **
            # We use the original subject IDs as the context for the mixed samples.
            inputs = {
                'sequence_input': X_seq_mix, 
                'static_input': X_static_mix, 
                'tabular_input': X_tabular_mix, 
            }
            outputs = {'main_output': y_mix, 'tof_gate': gate_target_mix}
            
            return inputs, outputs, sample_weights_mix

        # --- Return if not using Mixup ---
        inputs = {
            'sequence_input': Xb_seq, 
            'static_input': Xb_static, 
            'tabular_input': Xb_tabular, 
        }
        outputs = {'main_output': yb, 'tof_gate': gate_target}

        return inputs, outputs, sample_weights

    def on_epoch_end(self):
        """Shuffles indices at the end of each epoch."""
        np.random.shuffle(self.indices)

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super().__init__(); self.initial_learning_rate, self.decay_steps, self.warmup_steps, self.alpha = initial_learning_rate, decay_steps, warmup_steps, alpha
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps - warmup_steps, alpha)
    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        return tf.cond(step_float < self.warmup_steps, lambda: self.initial_learning_rate * (step_float / self.warmup_steps), lambda: self.cosine_decay(step_float - self.warmup_steps))
    def get_config(self): return {"initial_learning_rate": self.initial_learning_rate, "decay_steps": self.decay_steps, "warmup_steps": self.warmup_steps, "alpha": self.alpha}

class SWA(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch): super().__init__(); self.start_epoch, self.swa_weights, self.n_models = start_epoch, None, 0
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            if self.swa_weights is None: self.swa_weights = self.model.get_weights()
            else: self.swa_weights = [(s * self.n_models + w) / (self.n_models + 1) for s, w in zip(self.swa_weights, self.model.get_weights())]
            self.n_models += 1
            if epoch == self.params['epochs'] - 1: print("\nSetting final model weights to SWA average."); self.model.set_weights(self.swa_weights)