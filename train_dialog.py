import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import pickle

print("Loading dataset...")

inputs = []
responses = []

with open("dialogs.txt", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            inp, resp = line.strip().split("\t", 1)
            inputs.append(inp.lower())
            responses.append("<start> " + resp.lower() + " <end>")

print(f"Loaded {len(inputs)} dialog pairs.")

# Tokenizer
tokenizer = Tokenizer(filters='', oov_token="<OOV>")
tokenizer.fit_on_texts(inputs + responses)

input_seq = tokenizer.texts_to_sequences(inputs)
response_seq = tokenizer.texts_to_sequences(responses)

max_input_len = max(len(seq) for seq in input_seq)
max_target_len = max(len(seq) for seq in response_seq)

input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
response_seq = pad_sequences(response_seq, maxlen=max_target_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1

print("Vocab size:", vocab_size)

# Prepare decoder inputs and targets
decoder_input_data = response_seq[:, :-1]
decoder_target_data = response_seq[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, 128, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, 128, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training...")
model.fit(
    [input_seq, decoder_input_data],
    decoder_target_data,
    batch_size=32,
    epochs=100
)

model.save("dialog_model.keras")

with open("dialog_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("input_len.pkl", "wb") as f:
    pickle.dump(max_input_len, f)

with open("target_len.pkl", "wb") as f:
    pickle.dump(max_target_len, f)

print("✅ Training complete.")