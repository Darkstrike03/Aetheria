from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenize the cleaned conversations
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(cleaned_conversations)
sequences = tokenizer.texts_to_sequences(cleaned_conversations)

# Pad the sequences to the same length
maxlen = 100
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Split the data into training and validation sets
train_sequences, val_sequences = padded_sequences[:train_size], padded_sequences[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=vocab_size, activation="softmax"))

# Train the model
model.fit(train_sequences, train_labels, validation_data=(val_sequences, val_labels), epochs=10, batch_size=32)

# Save the trained model to a file
model.save("models/model.h5")