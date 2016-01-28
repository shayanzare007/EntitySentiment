# EntitySentiment
Entity level sentiment analysis for product reviews using deep learning

The objective of this project is to analyze Amazon reviews at the entity level: That is, looking at most most important aspects of a particular item, we would like to be able to predict the sentiment for each of these aspects on the same model. For more information about the methodology and theory please refer to:

http://cs224d.stanford.edu/reports/AhresY.pdf

In this code we implemented 4 models:
Recurrent Neural Network (RNN)
Bidirectionnal Recurrent Neural Network (BRNN)
Weighted Recurrent Neural Network (WRNN): weighting cost function to balance training data
Long Short Term Memory Neural Network (LSTM)
Bidirectionnal Long Short Term Memory Neural Network (BLSTM)

In order to install and run the current implementation, you need:
- Python 2.7
- Install the required packages summarized in requirement.txt

To run an example you simply type:
python RunRNN.py

However, we are unable to release the full data set we worked on. We hereby share parts of it that allow the models to run. It can be found on example_data folder.

