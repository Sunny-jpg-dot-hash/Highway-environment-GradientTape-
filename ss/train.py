import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 超參數設置
learning_rate = 0.001
epochs = 50
batch_size = 32

def load_training_data():
    # 加載訓練數據
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = to_categorical(train_dataset['label'])
    return train_data, train_label

def load_validation_data():
    # 加載驗證數據
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))  # 修改文件名
    valid_data = valid_dataset['data']
    valid_label = to_categorical(valid_dataset['label'])
    return valid_data, valid_label

def train_model():
    # 定義模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # 優化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 加載訓練數據
    train_data, train_label = load_training_data()
    
    # 自訂訓練循環
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_data), batch_size):
            x_batch = train_data[i:i + batch_size]
            y_batch = train_label[i:i + batch_size]
            
            # 使用 GradientTape 計算損失和梯度
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)
            
            # 計算梯度並更新參數
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(train_data) // batch_size)}")

    # 保存模型
    model.save('YOURMODEL.h5')

def evaluate_model():
    # 加載訓練好的模型
    model = tf.keras.models.load_model('YOURMODEL.h5')

    # 加載驗證數據
    valid_data, valid_label = load_validation_data()

    # 進行預測
    predictions = model.predict(valid_data, batch_size=batch_size)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(valid_label, axis=1)

    # 計算準確率
    accuracy = np.mean(true_labels == predicted_labels)
    print(f'Predicted labels: {predicted_labels}')
    print(f'True labels: {true_labels}')
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    # 訓練模型
    train_model()

    # 評估模型
    evaluate_model()
