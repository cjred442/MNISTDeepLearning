# MNISTDeepLearning
MNIST Number Predictor using Deep Learning.
<br />
<br />
Using an MNIST dataset provided by Kaggle, I have generated a Deep Learning model capable of predicting hand written numbers.

<h2>Data Set</h2>
Using a limited data set of 2000 MNIST samples was used, each sample was represented as a 28 x 28 array flattened to 784 values with the 
first value being the represented number (0-9). The the darkness of each pixel is repesented as a number 0-9. Each number can be represented
by matplotlib as below:
<p align="center">
<img src="https://gdurl.com/dIZj"/>
</p>
<h2>Model</h2>
The input array has a width of 784 (28 x 28) and a categorical output layer of 10 nodes (numbers 0 -9). The neural network devised includes
6 dense layers with 125 neurons, each layer employs the reLU actvation function. The output layer employs the softmax activation 
function. The optimization function used is ADAM and the loss function chosen is the categorical cross entropy used for multi-class 
classification. A split of 3:1 training to validation was used.

