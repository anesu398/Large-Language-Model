import numpy as np
import tensorflow as tf
from tensorflow import Keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Read the text data
with open('text_data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Preprocess the text data
text = text.lower()  # Convert text to lowercase
chars = sorted(list(set(text)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}
vocab_size = len(chars)

# Tokenize the text into input-output pairs
seq_length = 100
step = 3
input_seqs = []
output_chars = []
for i in range(0, len(text) - seq_length, step):
    input_seqs.append(text[i:i + seq_length])
    output_chars.append(text[i + seq_length])

# Vectorize the input sequences for original data
X = np.zeros((len(input_seqs), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(input_seqs), vocab_size), dtype=np.bool)
for i, seq in enumerate(input_seqs):
    for t, char in enumerate(seq):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[output_chars[i]]] = 1

# Data Augmentation
def augment_text(text, n_augmented_samples=1000, max_permutations=3):
    augmented_seqs = []
    for _ in range(n_augmented_samples):
        seq = random.choice(input_seqs)
        seq = list(seq)
        for _ in range(max_permutations):
            idx1, idx2 = random.sample(range(len(seq)), 2)
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
        augmented_seqs.append(''.join(seq))
    return augmented_seqs

# Augment the training data
augmented_input_seqs = augment_text(text)
augmented_output_chars = [text[i + seq_length] for i in range(len(augmented_input_seqs))]

# Vectorize the augmented input sequences
X_augmented = np.zeros((len(augmented_input_seqs), seq_length, vocab_size), dtype=np.bool)
y_augmented = np.zeros((len(augmented_input_seqs), vocab_size), dtype=np.bool)
for i, seq in enumerate(augmented_input_seqs):
    for t, char in enumerate(seq):
        X_augmented[i, t, char_to_index[char]] = 1
    y_augmented[i, char_to_index[augmented_output_chars[i]]] = 1

# Concatenate original and augmented data
X_combined = np.concatenate((X, X_augmented))
y_combined = np.concatenate((y, y_augmented))

# Shuffle the combined data
combined_data = list(zip(X_combined, y_combined))
random.shuffle(combined_data)
X_combined, y_combined = zip(*combined_data)
X_combined = np.array(X_combined)
y_combined = np.array(y_combined)

# Define the model architecture with dropout regularization
model = tf.keras.Sequential([
    LSTM(256, input_shape=(seq_length, vocab_size), return_sequences=True),
    Dropout(0.2),
    LSTM(256),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Define callbacks for monitoring and early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_combined, y_combined, batch_size=128, epochs=50, callbacks=[early_stopping])

# Save the model and vocabulary
model.save('language_model.h5')
np.save('vocab.npy', char_to_index)

# Fine-tuning with GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenized_text = tokenizer.batch_encode_plus(text, return_tensors="tf", padding=True, truncation=True, max_length=1024)["input_ids"]
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model_gpt2.compile(optimizer="adam", loss=loss_fn)
model_gpt2.fit(tokenized_text, tokenized_text, batch_size=8, epochs=3)
