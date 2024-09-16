Music Genre Classification Project 

This is the model for classifying musical genres that was trained on the GTZAN dataset, a publicly available dataset from Kaggle, by a deep convolutional neural network (CNN).

Ten genres—"blues," "classical," "country," "disco," "hiphop," "jazz," "metal," "pop," "reggae," and "rock"—can be classified by the model in that order.

The CNN model and training notebook are available for viewing in the Model_Training Folder. Run streamlit_app.py to see the application's demo.

Data Understanding
- Checking the samples in the dataset to see whether it is balanced or not.
-  Import IPython to play the audio data samples.
- E Extract and plot the spectrograms, which will be used as data for training the model.
  
Data Preprocessing
- First, define a function to load the audio data and extract spectrogram values. Chunk the audio into smaller pieces by adding overlap duration between chunks. Iterate through all chunk pieces and extract each chunk's spectrogram values. Add each value and its corresponding label into new lists and return.
- By using that function, extract values of audio samples we have from our GTZAN dataset.
- Split train, test data (80%, 20%), respectively.

Modelling
- The CNN model is built by a total 18 layers: 5 2D convolutional layers along with 'RELU' activation, 5 Max Pooling layers, 4 dropout layers, 1 flattening layer, and 3 dense layers. The last dense layer is an output layer with the 'Softmax' activation function.
- Trained the model with the batch size 64 & 40 epochs.

Evaluation
- Achieved an accuracy score of 99% on the training dataset and 91% on the testing dataset.
- For the evaluation metrics on testing data, model results achieved 91% accuracy, 91% precision, 91.5% recall, and 91% F1 score.
  
