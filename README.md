# Stock-Prediction-with-CNN-LSTM-Model

Combining the advantages of feature extraction of CNN model and sequence data analysis of LSTM model, this project applies the combined model CNN-LSTM to financial time series analysis and uses historical trading day data to predict the closing point of the next day.

At the same time, this project confirmed CNN-LSTM model has certain advantages in prediction accuracy compared with the single-structure CNN model, LSTM model and ANN model. 

---------

Using TuShare as a utility for crawling historical data of China stocks. 
 
This project takes the historical price of the CSI 300 Index as a research sample, conducts a practical effect test, and adjusts on a series of hyperparameters of the CNN-LSTM model, including the number of hidden layer neurons, initial learning step, dropout, the size of historical trading day window and the size of the convolution kernel.


Requirement
-----------
* Python 3.5
* TuShare 1.2.41
* Pandas 0.25.0
* Keras 2.2.5
* Numpy 1.16.4
* scikit-learn 0.21.2
* TensorFlow 1.14.0 (GPU version recommended)
