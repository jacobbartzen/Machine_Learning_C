## Machine Learning from Scratch

This project features two main code files, both coded in C: perceptron.c and neuralNetwork.c. neuralNetwork builds on perceptron.c by adding multiple layers, while perceptron.c is a single neuron.

## perceptron.c

This file has one perceptron. It is able to fit one linear line to the data provided. The user is able to adjust settings such as the learning rate, epochs, all data, and features such as early stopping and data normalization. The code tracks runtime, error, and prints all weights

## How a neuron works

In a neural network, a neuron takes in inputs and runs them through a function to produce a single output. The basic steps are:

1. Weight all inputs and add a bias
   - y = w1x1 + w2x2... + wnxn + b = ∑wx + b

2. Pass the output through an activation function. This model uses the ReLU activation function.
   - if (y < 0) y = 0

## Training

A neuron learns by adjusting its weights to reduce error. In a simple model, this can be done by updating each weight proportionally to the input value, the error, and the learning rate. For multi-layer neural networks, more advanced methods such as backpropagation are used.

In my example, weights are increased if the output was too low, and decreased if the output was too high proportionally to the input size and learning rate.

  weights[j] += LEARNING_RATE * eTotal * inputs[i][j];

## Neuron vs. Perceptron

While the file is called perceptron.c, there is one main difference between a perceptron and the neuron coded: A perceptron either outputs 1 or 0 (or sometimes 1 / -1). A perceptron was one of the very early approaches to machine learning. It aimed to mimic a human brain, where neurons either fire or don't fire - binary states. In my file perceptron.c, the neuron outputs a range of values.

## neuralNetwork.c

neuralNetwork.c is a full neural network. Using the array neuronLayers, the amount of layers and neurons in each layer can be easily adjusted. Featuee such as early stopping, dropout, and max weights are currently being added.

## How the neural network works

# Forward Pass

In the forward pass, the output is computed. Starting with the base layer of neurons, they each apply weights to the inputs, add a bias, and then apply the activation function. This is the same as in perceptron.c

After the first layer of neruons have each computed their output, those outputs are each sent to the next lager of neurons. This is repeated for as many layers as needed.

# Activation Function - Leaky ReLU

One major problem encoutered by many neural networks using the ReLU activation function is that neurons get stuck at 0, or "die". once a neuron goes into the negatives, the activation function ReLU sets its value to 0. During backpropogation, the weights for that neuron are not updated since it contributed nothing to the total error. In this way, the neuron is stuck at 0 for the remainder of training and testing.

To prevent this, Leaky ReLU still allows the weights of neurons that outputted nothing to still be updated, just at a smaller scale. This allows all neurons to stay alive and for the user to use higher learning rates without the risk of having a large amount of neurons die.

## Backpropogation

# Assigning Blame

In order to correct each neuron, it is necessary to find how much each neuron contributed to the total error. This is dine by startung with the total error of the system, and working back layer by layer to find how much each neuron contributed to the total error. The steps are shown below:

1. Calculate the total error of the system. This is the blame of the output neuron.
    Blame = eTotal

2. Multiply the blame of the neuron in the next layer by the weight connecting the current neuron and that output neuron. If the current neuron is connected to many neurons, add the result of wach multiplication
    Blame = 0
    For every neuron in next layer
        Blame += previousNeuronBlame * connectingWeight

3. Multiply the totalblane for that neuron by the deriative of the activation function - leaky ReLU.
    If neuron output < 0
        Blame *= 0.01

    Else
        Blame = Blame

If the activation function was ReLU and not leaky ReLU, the blame would be 0 if the neuron output was under 0. This would result in no change to that neurons weights, and the neuron would stay stuck outputting 0.

# Weight Updates

Now that each neuron has its blame assigned for how much it contributed to the final error, it is possible to update the weights and bias for each neuron.

1. Update bias based off of learningRate and Blame
  Bias += LEARNING_RATE * Blame

2. Update weights based off of learningRate, Blame, and the previous layers output
  Weight += LEARNING_RATE * Blame * PreviousLayerOutput
