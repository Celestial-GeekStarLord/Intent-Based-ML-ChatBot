import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Load everything
model = tf.keras.models.load_model("dialog_model.keras")

with open("dialog_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("input_len.pkl", "rb") as f:
    max_input_len = pickle.load(f)

with open("target_len.pkl", "rb") as f:
    max_target_len = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1

# Rebuild encoder model
encoder_inputs = model.input[0]
encoder_embedding = model.layers[2]
encoder_lstm = model.layers[3]
_, state_h_enc, state_c_enc = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Rebuild decoder model
decoder_inputs = model.input[1]
decoder_embedding = model.layers[4]
decoder_lstm = model.layers[5]
decoder_dense = model.layers[6]

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = decoder_embedding(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + [state_h2, state_c2]
)

def generate_response(input_text):
    seq = tokenizer.texts_to_sequences([input_text.lower()])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_input_len)

    states = encoder_model.predict(seq, verbose=0)

    target_seq = np.zeros((1,1))
    target_seq[0,0] = tokenizer.word_index["<start>"]

    stop = False
    decoded_sentence = []

    while not stop:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, "")

        if sampled_word == "<end>" or len(decoded_sentence) > max_target_len:
            stop = True
        else:
            decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index
        states = [h, c]

    return " ".join(decoded_sentence)


print("Chatbot ready! Type quit to exit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = generate_response(user_input)
    print("Bot:", response)