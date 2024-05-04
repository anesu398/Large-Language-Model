from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

# Define the length of the input sequence
seq_length = 100
app = Flask(__name__)

# Load the trained model and vocabulary
model = tf.keras.models.load_model('language_model.h5')
vocab = np.load('vocab.npy', allow_pickle=True).item()

# Function to generate text
def generate_text(seed_text, length=100):
    generated_text = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(vocab)))
        for t, char in enumerate(seed_text):
            if char in vocab:
                x_pred[0, t, vocab[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(vocab), p=preds)
        next_char = [char for char, index in vocab.items() if index == next_index][0]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form['seed_text']
    generated_text = generate_text(seed_text)
    return render_template('index.html', seed_text=seed_text, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
