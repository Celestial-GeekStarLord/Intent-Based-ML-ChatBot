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

# Tokenizer
tokenizer = Tokenizer(filters='', oov_token="<OOV>")
tokenizer.fit_on_texts(inputs + responses)

input_seq = tokenizer.texts_to_sequences(inputs)
target_seq = tokenizer.texts_to_sequences(responses)

max_input_len = max(len(s) for s in input_seq)
max_target_len = max(len(s) for s in target_seq)

input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding="post")
target_seq = pad_sequences(target_seq, maxlen=max_target_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1

decoder_input_data = target_seq[:, :-1]
decoder_target_data = target_seq[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Encoder
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(vocab_size, 128, mask_zero=True, name="encoder_embedding")(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = Embedding(vocab_size, 128, mask_zero=True, name="decoder_embedding")
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(256, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    [input_seq, decoder_input_data],
    decoder_target_data,
    batch_size=32,
    epochs=100
)

# -------- INFERENCE MODELS --------

# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference setup
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)

decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# Save everything
model.save("Model_Output/dialog_model.keras")
encoder_model.save("Model_Output/encoder_model.keras")
decoder_model.save("Model_Output/decoder_model.keras")

with open("Model_Output/dialog_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("Model_Output/input_len.pkl", "wb") as f:
    pickle.dump(max_input_len, f)

with open("Model_Output/target_len.pkl", "wb") as f:
    pickle.dump(max_target_len, f)

print("✅ Training complete.")