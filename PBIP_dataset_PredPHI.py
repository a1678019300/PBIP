import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import keras.applications
from keras.layers import *
from keras.models import *
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    average_precision_score
)
from keras.optimizers import adam_v2
from keras.callbacks import Callback


NUM_LAYER = 4
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
EPOCHS = 200
FILTERS = 32
NUM_UNIT = 64
KFOLDS = 10
RANDOM_SEED = 42


def obtain_features(data_df, phage_unirep_dir, host_unirep_dir, file_suffix='.txt'):
    phage_features = []
    host_features = []
    labels = []

    for _, row in data_df.iterrows():
        phage_id = row['phage']
        host_id = row['host']
        label = row['label']

        phage_file = f"{phage_unirep_dir}/{phage_id}{file_suffix}"
        phage_fea = np.loadtxt(phage_file)

        host_file = f"{host_unirep_dir}/{host_id}{file_suffix}"
        host_fea = np.loadtxt(host_file)

        phage_features.append(phage_fea)
        host_features.append(host_fea)
        labels.append(label)

    return np.array(phage_features), np.array(host_features), np.array(labels)


train_csv_dir = 'PredPHI_train_set.csv'
test_csv_dir = 'PredPHI_test_set.csv'


phage_unirep_dir = "data/PredPHI/phage"
host_unirep_dir = "data/PredPHI/host"


fold_result_dir = '10_fold_result/dataset_PredPHI'
test_result_dir = 'test_result/dataset_PredPHI'
model_result_dir = 'model/dataset_PredPHI'


os.makedirs(fold_result_dir, exist_ok=True)
os.makedirs(test_result_dir, exist_ok=True)
os.makedirs(model_result_dir, exist_ok=True)


def prepare_features():
    train_df = pd.read_csv(train_csv_dir)
    test_df = pd.read_csv(test_csv_dir)
    X_train_phage, X_train_host, y_train = obtain_features(
        train_df, phage_unirep_dir, host_unirep_dir
    )
    X_test_phage, X_test_host, y_test = obtain_features(
        test_df, phage_unirep_dir, host_unirep_dir
    )
    input_dim = 1900
    return X_train_phage, X_train_host, y_train, X_test_phage, X_test_host, y_test, input_dim


def attention_3d_block(inputs, name):
    TIME_STEPS = inputs.shape.as_list()[1]
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def pbip_model(input_dim):
    kernel_size = 3
    pooling_size = 2
    p_in = Input(shape=(input_dim,))
    p = Reshape((input_dim, 1))(p_in)
    p = Conv1D(filters=FILTERS, kernel_size=kernel_size, padding='same', activation='relu')(p)
    p = MaxPooling1D(pool_size=pooling_size)(p)

    if NUM_LAYER >= 2:
        p = Conv1D(filters=FILTERS * 2, kernel_size=kernel_size, padding='same', activation='relu')(p)
        p = MaxPooling1D(pool_size=pooling_size)(p)
    if NUM_LAYER >= 3:
        p = Conv1D(filters=FILTERS * 4, kernel_size=kernel_size, padding='same', activation='relu')(p)
        p = MaxPooling1D(pool_size=pooling_size)(p)
    if NUM_LAYER >= 4:
        p = Conv1D(filters=FILTERS * 8, kernel_size=kernel_size, padding='same', activation='relu')(p)
        p = MaxPooling1D(pool_size=pooling_size)(p)

    p = Dropout(0.5)(p)
    p = Model(inputs=p_in, outputs=p)
    b_in = Input(shape=(input_dim,))
    b = Reshape((input_dim, 1))(b_in)
    b = Conv1D(filters=FILTERS, kernel_size=kernel_size, padding='same', activation='relu')(b)
    b = MaxPooling1D(pool_size=pooling_size)(b)

    if NUM_LAYER >= 2:
        b = Conv1D(filters=FILTERS * 2, kernel_size=kernel_size, padding='same', activation='relu')(b)
        b = MaxPooling1D(pool_size=pooling_size)(b)
    if NUM_LAYER >= 3:
        b = Conv1D(filters=FILTERS * 4, kernel_size=kernel_size, padding='same', activation='relu')(b)
        b = MaxPooling1D(pool_size=pooling_size)(b)
    if NUM_LAYER >= 4:
        b = Conv1D(filters=FILTERS * 8, kernel_size=kernel_size, padding='same', activation='relu')(b)
        b = MaxPooling1D(pool_size=pooling_size)(b)

    b = Dropout(0.5)(b)
    b = Model(inputs=b_in, outputs=b)
    p_gru = Bidirectional(GRU(NUM_UNIT, return_sequences=True))(p.output)
    p_gru = attention_3d_block(p_gru, 'attention_phage')

    b_gru = Bidirectional(GRU(NUM_UNIT, return_sequences=True))(b.output)
    b_gru = attention_3d_block(b_gru, 'attention_bac')

    merge_layer = Concatenate(axis=1)([p_gru, b_gru])
    fl = Flatten()(merge_layer)
    bn = BatchNormalization()(fl)
    dt = Dropout(0.5)(bn)
    dt = Dense(64)(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    output = Dense(1, activation='sigmoid')(dt)

    return Model(inputs=[p_in, b_in], outputs=output)


def calculate_metrics(y_true, y_pred_prob, th=0.5):
    y_pred = (y_pred_prob > th).astype(int)

    # 基础指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    sensitivity = recall_score(y_true, y_pred, average='macro')
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    if len(np.unique(y_true)) <= 1:
        auc = 0.0
        aupr = 0.0
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return [acc, precision, sensitivity, specificity, f1, mcc, auc, aupr]


def main():
    train_phage, train_host, train_labels, test_phage, test_host, test_labels, input_dim = prepare_features()

    print("\n===== 10-fold-cv =====")
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_phage)):
        print(f"\n-----  {fold + 1}/{KFOLDS} fold -----")

        X_p_train_raw, X_p_val_raw = train_phage[train_idx], train_phage[val_idx]
        X_b_train_raw, X_b_val_raw = train_host[train_idx], train_host[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        scaler_p = StandardScaler()
        scaler_b = StandardScaler()
        X_p_train = scaler_p.fit_transform(X_p_train_raw)
        X_b_train = scaler_b.fit_transform(X_b_train_raw)
        X_p_val = scaler_p.transform(X_p_val_raw)
        X_b_val = scaler_b.transform(X_b_val_raw)

        X_p_train_resampled = X_p_train
        X_b_train_resampled = X_b_train
        y_train_resampled = y_train

        model = pbip_model(input_dim)
        adam = adam_v2.Adam(learning_rate=LEARNING_RATE, amsgrad=True, epsilon=1e-6)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

        model.fit(
            x=[X_p_train_resampled, X_b_train_resampled],
            y=y_train_resampled,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([X_p_val, X_b_val], y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
                )
            ]
        )

        y_val_prob = model.predict([X_p_val, X_b_val]).flatten()
        val_metrics = calculate_metrics(y_val, y_val_prob)
        fold_metrics.append(val_metrics)

        metrics_names = ["Accuracy", "Precision", "Sensitivity", "Specificity",
                         "F1-score", "MCC", "AUC", "AUPR"]
        print(f" {fold + 1} metrics：")
        for name, val in zip(metrics_names, val_metrics):
            print(f"{name}: {val:.4f}")

        fold_pred_path = os.path.join(fold_result_dir, f"fold_{fold}_pred.csv")
        np.savetxt(fold_pred_path,
                   np.column_stack((y_val, y_val_prob)),
                   delimiter=',',
                   header='true_label,pred_prob',
                   comments='')

    fold_metrics = np.array(fold_metrics)
    avg_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)

    fold_summary_path = os.path.join(fold_result_dir, "fold_summary.csv")
    with open(fold_summary_path, 'w') as f:
        f.write("metric,average,std\n")
        metrics_names = ["Accuracy", "Precision", "Sensitivity", "Specificity",
                         "F1-score", "MCC", "AUC", "AUPR"]
        for name, avg, std in zip(metrics_names, avg_metrics, std_metrics):
            f.write(f"{name},{avg:.4f},{std:.4f}\n")

    print("\n===== 10折交叉验证汇总 =====")
    for name, avg, std in zip(metrics_names, avg_metrics, std_metrics):
        print(f"{name}: {avg:.4f} ± {std:.4f}")

    print("\n===== 10-fold-cv results====")
    scaler_p_final = StandardScaler()
    scaler_b_final = StandardScaler()
    train_phage_scaled = scaler_p_final.fit_transform(train_phage)
    train_host_scaled = scaler_b_final.fit_transform(train_host)
    test_phage_scaled = scaler_p_final.transform(test_phage)
    test_host_scaled = scaler_b_final.transform(test_host)

    train_phage_resampled = train_phage_scaled
    train_host_resampled = train_host_scaled
    train_labels_resampled = train_labels

    final_model = pbip_model(input_dim)
    adam = adam_v2.Adam(learning_rate=LEARNING_RATE, amsgrad=True, epsilon=1e-6)
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    final_model.fit(
        x=[train_phage_resampled, train_host_resampled],
        y=train_labels_resampled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
            )
        ]
    )

    final_model_path = os.path.join(model_result_dir, "final_model.h5")
    final_model.save(final_model_path)

    y_test_prob = final_model.predict([test_phage_scaled, test_host_scaled]).flatten()
    test_metrics = calculate_metrics(test_labels, y_test_prob)

    test_pred_path = os.path.join(test_result_dir, "test_pred.csv")
    np.savetxt(test_pred_path,
               np.column_stack((test_labels, y_test_prob)),
               delimiter=',',
               header='true_label,pred_prob',
               comments='')

    test_metrics_path = os.path.join(test_result_dir, "test_metrics.csv")
    with open(test_metrics_path, 'w') as f:
        f.write("metric,value\n")
        metrics_names = ["Accuracy", "Precision", "Sensitivity", "Specificity",
                         "F1-score", "MCC", "AUC", "AUPR"]
        for name, val in zip(metrics_names, test_metrics):
            f.write(f"{name},{val:.4f}\n")

    print("\n===== test results =====")
    for name, val in zip(metrics_names, test_metrics):
        print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()