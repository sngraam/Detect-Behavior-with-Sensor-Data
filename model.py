# model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GRU, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, RepeatVector, Embedding
)
from tensorflow.keras.regularizers import l2

def time_sum(x):
    """Sums features across the time dimension."""
    return tf.reduce_sum(x, axis=1)

def squeeze_last_axis(x):
    """Removes the last dimension of a tensor."""
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    """Adds a new dimension at the end of a tensor."""
    return tf.expand_dims(x, axis=-1)


def se_block(x, reduction=8, wd=1e-4):
    """Squeeze-and-Excitation block."""
    num_channels = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(num_channels // reduction, activation='relu', kernel_regularizer=l2(wd))(se)
    se = Dense(num_channels, activation='sigmoid', kernel_regularizer=l2(wd))(se)
    se = Reshape((1, num_channels))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """A residual block with two 1D convolutions and a Squeeze-and-Excitation block."""
    shortcut = x

    # First Conv Layer
    x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Conv Layer
    x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Squeeze-and-Excitation
    x = se_block(x, wd=wd)

    # Add shortcut connection
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    # Pooling and Dropout
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs, wd=1e-4):
    """Attention mechanism to weigh the importance of different time steps."""
    score = Dense(1, activation='tanh', kernel_regularizer=l2(wd))(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

# Main Model Architecture 
def build_monster_branch_model(pad_len, imu_dim, thm_dim, tof_dim, static_dim, tabular_dim, n_classes, wd=3e-4):
    """
    Builds a deep multi-branch model for sensor-based gesture recognition.
    """
    sequence_inp = Input(shape=(pad_len, imu_dim + thm_dim + tof_dim), name='sequence_input')
    static_inp = Input(shape=(static_dim,), name='static_input')
    tabular_inp = Input(shape=(tabular_dim,), name='tabular_input')

    # SENSOR DATA SPLITTING ---
    imu = Lambda(lambda t: t[:, :, :imu_dim], name='split_imu')(sequence_inp)
    thm = Lambda(lambda t: t[:, :, imu_dim:imu_dim + thm_dim], name='split_thm')(sequence_inp)
    tof = Lambda(lambda t: t[:, :, imu_dim + thm_dim:], name='split_tof')(sequence_inp)

    # Separate IMU components for initial processing
    acc_data = Lambda(lambda t: t[:, :, :3])(imu)   
    gyro_data = Lambda(lambda t: t[:, :, 3:6])(imu) 
    other_imu_data = Lambda(lambda t: t[:, :, 6:])(imu)

    # Initial lightweight Conv layers for each component
    acc_feat = Conv1D(32, 5, padding='same', use_bias=False, kernel_regularizer=l2(wd))(acc_data)
    acc_feat = BatchNormalization()(acc_feat); acc_feat = Activation('relu')(acc_feat)

    gyro_feat = Conv1D(32, 5, padding='same', use_bias=False, kernel_regularizer=l2(wd))(gyro_data)
    gyro_feat = BatchNormalization()(gyro_feat); gyro_feat = Activation('relu')(gyro_feat)

    other_imu_feat = Conv1D(64, 5, padding='same', use_bias=False, kernel_regularizer=l2(wd))(other_imu_data)
    other_imu_feat = BatchNormalization()(other_imu_feat); other_imu_feat = Activation('relu')(other_imu_feat)
    
    # Merge and process through deeper residual blocks
    imu_merged = Concatenate()([acc_feat, gyro_feat, other_imu_feat])
    imu_merged = MaxPooling1D(2)(imu_merged); imu_merged = Dropout(0.2)(imu_merged)

    imu_head = residual_se_cnn_block(imu_merged, 96, 3, drop=0.2, wd=wd)
    imu_head = residual_se_cnn_block(imu_head, 128, 5, drop=0.2, wd=wd)

    # ToF data is already aggregated per sensor (mean, std, etc.). Let's process it as one block.
    tof_cnn = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    tof_cnn = BatchNormalization()(tof_cnn); tof_cnn = Activation('relu')(tof_cnn)
    tof_cnn = MaxPooling1D(2)(tof_cnn); tof_cnn = Dropout(0.2)(tof_cnn)
    
    # Gating Mechanism ---
    gate_input = GlobalAveragePooling1D()(tof) # Use the raw ToF input for the gate
    gate_dense = Dense(16, activation='relu')(gate_input)
    gate = Dense(1, activation='sigmoid', name='tof_gate')(gate_dense)
    
    # We apply the gate to the first layer of ToF processing
    tof_gated = Multiply()([tof_cnn, gate])

    tof_head = residual_se_cnn_block(tof_gated, 96, 3, drop=0.2, wd=wd)
    tof_head = residual_se_cnn_block(tof_head, 128, 5, drop=0.2, wd=wd)

    thm_cnn = Conv1D(32, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(thm)
    thm_cnn = BatchNormalization()(thm_cnn); thm_cnn = Activation('relu')(thm_cnn)
    thm_cnn = MaxPooling1D(2)(thm_cnn); thm_cnn = Dropout(0.2)(thm_cnn)
    
    thm_head = residual_se_cnn_block(thm_cnn, 64, 3, drop=0.2, wd=wd)
    thm_head = residual_se_cnn_block(thm_head, 96, 5, drop=0.2, wd=wd)

    static_branch = Dense(64, activation='relu')(static_inp)
    static_branch = Dense(32, activation='relu')(static_branch)

    target_time_steps = imu_head.shape[1]
    static_repeated = RepeatVector(target_time_steps)(static_branch)

    # Merge all the processed sequence features together
    merged_sequence = Concatenate()([imu_head, tof_head, thm_head, static_repeated])

    # Bidirectional RNN layers to capture temporal patterns
    rnn_out = Bidirectional(LSTM(192, return_sequences=True, kernel_regularizer=l2(wd)))(merged_sequence)
    rnn_out = Dropout(0.4)(rnn_out)

    # Attention to summarize RNN output into a single vector
    attention_vector = attention_layer(rnn_out, wd=wd)
    
    # Tabular branch for aggregated features
    tabular_branch = Dense(128, activation='relu')(tabular_inp)
    tabular_branch = BatchNormalization()(tabular_branch)
    tabular_branch = Dropout(0.4)(tabular_branch)
    tabular_branch = Dense(64, activation='relu')(tabular_branch)

    # Combine the time-series summary (attention) with the tabular features
    final_combined = Concatenate()([attention_vector, tabular_branch])

    # Final MLP Classifier
    x = Dense(256, use_bias=False, kernel_regularizer=l2(wd))(final_combined)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    main_output = Dense(n_classes, activation='softmax', name='main_output')(x)

    model = Model(
        inputs={'sequence_input': sequence_inp, 'static_input': static_inp, 'tabular_input': tabular_inp},
        outputs={'main_output': main_output, 'tof_gate': gate}
    )
    return model 
