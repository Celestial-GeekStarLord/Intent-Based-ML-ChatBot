import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import mixed_precision
import pickle

mixed_precision.set_global_policy('mixed_float16')

# Load dialog dataset
inputs = []
responses = []

with open("dialogs.txt", encoding="utf-8") as f:
    for line in f:
        if "," in line:
            inp, resp = line.strip().split(",", 1)
            inputs.append(inp.lower())
            responses.append(resp.lower())

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + responses)

input_seq = tokenizer.texts_to_sequences(inputs)
response_seq = tokenizer.texts_to_sequences(responses)

max_len = max(max(len(s) for s in input_seq),
              max(len(s) for s in response_seq))

input_seq = pad_sequences(input_seq, maxlen=max_len, padding="post")
response_seq = pad_sequences(response_seq, maxlen=max_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1

# Build model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(128),
    Dense(vocab_size, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train using first token of response as label (simple approach)
y = response_seq[:,0]

model.fit(input_seq, y, epochs=10)

# Save files
model.save("dialog_model.keras")

with open("dialog_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("Dialog model trained!")