Music Genre Classification Project 

This is the music genre classification model which was trained by deep convolutional neural network (CNN) using GTZAN dataset( a public dataset from Kaggle).
The model can classify 10 types of genre, 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae',and 'rock'  accoringly.
You can check the CNN model and training notebook in Model_Training Folder. The Demo of the app can be accessed by running streamlit_app.py.

Data Understanding
- Checking the samples in dataset whether it is balanced or not
- Import IPythin to play the audio data samples
- Extract and plot the spectrograms which will be used as a data for training the model

Data Preprocessing
- First, define function to load the audio data and extract spectrograms values. Chunk the audio into smallers pieces by adding overlap duration between chunks. Iterate through all chunk pieces and extract each chunk spectrograms values. Add each values and its corresponding label into new lists and return.
- By using that function, extract values of audio samples we have from our GTZAN dataset.
- Split train, test data (80%,20%) respectively.


  
