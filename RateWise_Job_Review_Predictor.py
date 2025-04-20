import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

#------------------------------------------------------------------#

# Natural Language Processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------#

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#------------------------------------------------------------------#

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#------------------------------------------------------------------#

#Neural Network
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#--------------------------------------------#
# LOAD & PREPROCESS DATA
#--------------------------------------------#

df = pd.read_csv(r"C:\Users\aisha\OneDrive\Desktop\BTT AI\glassdoor_reviews.csv\glassdoor_reviews.csv")
df.head()

df['review_text'] = df[['headline', 'pros', 'cons']].fillna('').agg(' '.join, axis=1)

df.drop(columns=['headline', 'pros', 'cons'], inplace=True)

features = ['overall_rating', 'review_text', 'work_life_balance',
                        'culture_values', 'diversity_inclusion',
                        'career_opp', 'comp_benefits', 'senior_mgmt']

df = df[features]

print(df.dtypes)
print(df.shape)

df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')
df['work_life_balance'] = pd.to_numeric(df['work_life_balance'], errors='coerce')
df['culture_values'] = pd.to_numeric(df['culture_values'], errors='coerce')
df['diversity_inclusion'] = pd.to_numeric(df['diversity_inclusion'], errors='coerce')
df['career_opp'] = pd.to_numeric(df['career_opp'], errors='coerce')
df['comp_benefits'] = pd.to_numeric(df['comp_benefits'], errors='coerce')
df['senior_mgmt'] = pd.to_numeric(df['senior_mgmt'], errors='coerce')

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(subset=['overall_rating', 'work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt'], inplace=True)

df['overall_rating'] = df['overall_rating'].astype(int)
df['work_life_balance'] = df['work_life_balance'].astype(int)
df['culture_values'] = df['culture_values'].astype(int)
df['diversity_inclusion'] = df['diversity_inclusion'].astype(int)
df['career_opp'] = df['career_opp'].astype(int)
df['comp_benefits'] = df['comp_benefits'].astype(int)
df['senior_mgmt'] = df['senior_mgmt'].astype(int)

# df['overall_rating'] = df['overall_rating'].fillna(df['overall_rating'].median()).astype(int)
# df['work_life_balance'] = df['work_life_balance'].fillna(df['work_life_balance'].median()).astype(int)
# df['culture_values'] = df['culture_values'].fillna(df['culture_values'].median()).astype(int)
# df['diversity_inclusion'] = df['diversity_inclusion'].fillna(df['diversity_inclusion'].median()).astype(int)
# df['career_opp'] = df['career_opp'].fillna(df['career_opp'].median()).astype(int)
# df['comp_benefits'] = df['comp_benefits'].fillna(df['comp_benefits'].median()).astype(int)
# df['senior_mgmt'] = df['senior_mgmt'].fillna(df['senior_mgmt'].median()).astype(int)

print(df.dtypes)
print(df.shape)

print(df.isnull().sum())

print(df.dtypes)

#--------------------------------------------#
# TRAIN-TEST SPLIT
#--------------------------------------------#

y = df['overall_rating']
X = df.drop(columns=['overall_rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)

y_train_adjusted = y_train_encoded + 1

unique_adjusted_labels = np.unique(y_train_adjusted)
print(f"Unique adjusted labels: {unique_adjusted_labels}")

num_classes = len(unique_adjusted_labels)  # Should be 5
y_train = to_categorical(y_train_adjusted - 1, num_classes=num_classes)

y_test_encoded = label_encoder.transform(y_test)
y_test_adjusted = y_test_encoded + 1
y_test = to_categorical(y_test_adjusted - 1, num_classes=num_classes)

#--------------------------------------------#
# TF-IDF VECTORIZATION
#--------------------------------------------#

tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(X_train['review_text'])

X_train_tfidf = tfidf_vectorizer.transform(X_train['review_text'])

X_test_tfidf = tfidf_vectorizer.transform(X_test['review_text'])

print('X_train_tfidf shape:', X_train_tfidf.shape)
print('X_test_tfidf shape:', X_test_tfidf.shape)

vocabulary_size = len(tfidf_vectorizer.vocabulary_)

print(vocabulary_size)

#--------------------------------------------#
# DEFINE NEURAL NETWORK MODEL
#--------------------------------------------#

input_shape = X_train_tfidf.shape[1]

nn_model = Sequential()

nn_model.add(InputLayer(input_shape=(input_shape,)))

nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(5, activation='softmax'))

nn_model.summary()

sgd_optimizer = SGD(learning_rate=0.1)

loss_fn = CategoricalCrossentropy(from_logits=False)

nn_model.compile(optimizer=sgd_optimizer,
                 loss=loss_fn,
                 metrics=['accuracy'])

#--------------------------------------------#
# TRAIN MODEL
#--------------------------------------------#

class ProgBarLoggerNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs: int, every_n: int = 50):
        super(ProgBarLoggerNEpochs, self).__init__()
        self.num_epochs = num_epochs
        self.every_n = every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)
                      for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))


t0 = time.time()

X_train_tfidf_array = X_train_tfidf.toarray()

num_epochs = 20

sample_size = 45000  # Adjust as necessary
X_train_tfidf_subset = X_train_tfidf_array[:sample_size]
y_train_subset = y_train[:sample_size]

history = nn_model.fit(
    X_train_tfidf_subset,
    y_train_subset,
    epochs=num_epochs,
    verbose=1,
    validation_split=0.2,
    callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=5)]
)

# Stop time
t1 = time.time()

# Print elapsed time
print('Elapsed time: %.2fs' % (t1 - t0))

history.history.keys()

#--------------------------------------------#
# PLOT TRAINING HISTORY
#--------------------------------------------#

# Plot training and validation loss
plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot training and validation accuracy
plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#--------------------------------------------#
# EVALUATION ON TEST SET
#--------------------------------------------#

loss, accuracy = nn_model.evaluate(X_test_tfidf.toarray(), y_test)

print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))

probability_predictions = nn_model.predict(X_test_tfidf.toarray())

y_test_array = y_test.argmax(axis=1)  # Convert one-hot encoded y_test back to labels

print("Predictions for the first 20 examples:")

for i in range(20):
    predicted_label = np.argmax(probability_predictions[i])
    actual_label = y_test_array[i]

    probability = probability_predictions[i][predicted_label]

    print(f"Example {i + 1}: Probability = {probability:.4f}, Predicted Label = {predicted_label + 1}, Actual Label = {actual_label + 1}")

#--------------------------------------------#
# DISPLAY SAMPLE REVIEWS & RATINGS
#--------------------------------------------#

num_reviews_to_display = 10

num_reviews_to_display = min(num_reviews_to_display, len(X_test))

review_texts = df.loc[X_test.index, 'review_text'].tolist()

for i in range(num_reviews_to_display):
    sample_index = i

 review_text = review_texts[sample_index]
    predicted_probabilities = probability_predictions[sample_index]
    predicted_rating = np.argmax(predicted_probabilities) + 1
    actual_rating = np.argmax(y_test[sample_index]) + 1

    print(f'Review #{sample_index + 1}:\n')
    print(review_text)
    print(f'\nPredicted Rating: {predicted_rating}\n')
    print(f'Actual Rating: {actual_rating}\n')
    print('-' * 80)

#--------------------------------------------#
# CONFUSION MATRIX
#--------------------------------------------#

predicted_labels = np.argmax(probability_predictions, axis=1) + 1

actual_labels = np.argmax(y_test, axis=1) + 1  # Convert to 1-based index

conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=[1, 2, 3, 4, 5])

print("Confusion Matrix:")
print(conf_matrix)

predicted_labels = np.argmax(probability_predictions, axis=1) + 1

actual_labels = np.argmax(y_test, axis=1) + 1

conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=[1, 2, 3, 4, 5])

row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_percentage = conf_matrix.astype(float) / row_sums
conf_matrix_percentage = np.nan_to_num(conf_matrix_percentage)

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_percentage, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label='Percentage')

for i in range(conf_matrix_percentage.shape[0]):
    for j in range(conf_matrix_percentage.shape[1]):
        plt.text(j, i, f'{conf_matrix_percentage[i, j]:.2f}',
                 ha='center', va='center', color='black', fontsize=12)

plt.xlabel('Predicted Rating', fontsize=14)
plt.ylabel('Actual Rating', fontsize=14)
plt.title('Confusion Matrix Percentage Heatmap', fontsize=16)
plt.xticks(ticks=np.arange(5), labels=[1, 2, 3, 4, 5], fontsize=12)
plt.yticks(ticks=np.arange(5), labels=[1, 2, 3, 4, 5], fontsize=12)
plt.grid(False)  # Disable gridlines in the plot
plt.show()

