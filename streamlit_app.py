import streamlit as st
import tensorflow as tf
import librosa
import numpy as np


def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 6  # seconds
    overlap_duration = 3  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        # Extract the chunk of audio
        chunk = audio_data[start:end]

        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(X_test, model):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

st.title(" Music Genre Classification Project...")
st.subheader("By Team Music Geeks")

uploaded_file = st.file_uploader("Upload an audio file.")


if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = True

st.button('Predict The Genre..', on_click=click_button)


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mpeg", loop=True)

    user_input = load_and_preprocess_data(uploaded_file)
    cnn_model = tf.keras.models.load_model('Model_Training/final_cnn_model.h5')
    
    if st.session_state.button:
        prediction = model_prediction(user_input, cnn_model)
        st.write('The Music you provided belong to the genre of',classes[prediction])
        
        
        sentiment_mapping = ["one", "two", "three", "four", "five"]
        selected = st.feedback("stars")
        if selected is not None:
            st.markdown(f"Thanks You for giving us {sentiment_mapping[selected]} star(s).")

        st.session_state.button = False