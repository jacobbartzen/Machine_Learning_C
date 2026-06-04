<h1 align="center">Machine Learning Library in C</h1>

# Overview

This repository features a machine learning library coded fully in C. All network parameters such as data, architecture, activation function, optimizers, and training hyperparameters are fully adjustable to the user.

| File | Overview |
| :---- | :---- |
| perceptron.c | Just one neuron basic code and layout \- Not part of the library. To run: `gcc perceptron.c -o main.exe` `.\main.exe` |
| neuralNetwork.c | All library function body code. |
| neuralNetwork.h | Structs and function prototypes |
| main.c | Main File \- Provides example outline and code for functions and features |
| networkWeights.bin | Binary file to store network weights for saveWeights() function. Will automatically be cleared and rewritten whenever function saveWeights() runs |
| oneFileNetwork.c | All of the library code in one file. Good for experimenting with the basic function level code, debugging quickly, or learning how the library works.To Run: `gcc oneFileNetwork.c -o main.exe` `.\main.exe` |

# Quickstart

Follow these steps to install and run the code from the library

1. In a new folder place the files:  
   1. neuralNetwork.c  
   2. neuralNetwork.h  
   3. Main.c  
   4. networkWeights.bin

2. If a C compiler is not currently installed, install a compiler. See the VS code guide on installing a C compiler for more information:  
   1. [https://code.visualstudio.com/docs/cpp/config-mingw](https://code.visualstudio.com/docs/cpp/config-mingw)

3. In a text editor such as VS code open the terminal, navigate to the folder containing the code files, and run the command:  
   `gcc main.c neuralNetwork.c -o main -lm`

4. Adjust hyperparameters such as activation function, optimizers, learning rate, epochs, and data in the file main.c. For more information see the section on hyperparameters.

## Common Definitions

**Parameter**: 

1. A value in the neural network not directly adjusted by the user such as weights or biases  
2. A variable passed to a function

**Hyperparameter**: A value in the neural network directly adjusted by the user such as learning rate

**Neuron**: A function that takes inputs and produces an output value

**Layer**: A series of parallel neurons used to build a neural network

**Input Layer**: The layer of a neural network that takes in the raw data as inputs

**Hidden Layer**: The layer(s) of a neural network that take in other neurons values as inputs

**Output Layer**: The final layer of a neural network that produces the final output

**Neural Network**: A type of machine learning model using interconnected layers of neurons to predict data

# Perceptron.c Overview

perceptron.c is a self-contained code file with just one neuron. It is able to fit one linear line through the data, and uses the equation for the line in order to generalize to new data. This can serve as a basic introduction to how neural networks work at a larger scale and adjust hyperparameters more smoothly without the exploding gradients and large error fluctuations that can occur on larger neural networks.

## How a neuron works

In a neural network, a neuron takes inputs and runs them through a function to produce a single output. Z is the output calculated before the activation function, and A is the post-activation function value. The basic steps are:

1. Weight all inputs and add a bias  
   * Z \= w1x1 \+ w2x2... \+ wnxn \+ b \= ∑wx \+ b

2. Pass the output through an activation function. This is not needed for one singular perceptron, but it is included in this file for demonstration purposes.   
   * A \= f(Z)

A neuron learns by adjusting its weights and biases to reduce error. With only one neuron, the equation for this is:

totalError \= expectedOutput \- calculatedOutput;

Bias \+= totalError \* LearningRate;

For each weight in the network:  
Weight\[i\] \+= totalError \* LearningRate \* X\[i\];

This process is repeated for each different datapoint, and then for each epoch. This process takes time to allow for the weights to settle at values that produce the lowest error possible, or the minimum of the function.

Neuron vs. Perceptron  
While the file is called perceptron.c, there is one main difference between a perceptron and the neuron coded: A perceptron either outputs binary states, typically either 0/1 or \-1/1. A perceptron was one of the very early approaches to machine learning and it aimed to mimic a human brain, where neurons either fire or don't fire. In my file perceptron.c, the neuron outputs a range of values. To turn this into a typical perceptron, a line would need to be added to change the range to binary states such as:  
	`Output = (A > 0.5) ? 0 : 1;`

# Neural Networks

In a neural network, layers of many neurons are connected, typically into dense layers where each neuron from the previous layer is connected to each neuron on the next layer. The general process of creating and training a neural network can be outlined below:

1. Allocate arrays for all of the parameters including weights, biases, velocity, blame, and individual values.

2. Set the weights to small, random values. The library uses HE initialization, which scales the random values by sqrt(2 / previousLayerNeurons). 

3. Normalize all inputs and labels. A typical value would be between 0 and 1\.

4. Training loop:  
   1. Forward Pass: Inputs are passed into the network and an output is produced  
   2. Backwards Pass: Backpropagation is used to find how much each weight contributed to the total error, and weights are adjusted accordingly.

5. Testing: New data that was not used for training is fed into the network to test how well the model can generalize to unseen data.

## Memory Allocation

The first step in creating a neural network is allocating the memory. Neural networks take a very large amount of storage, and storage space grows exponentially with the size of the network. For example, a network with an architecture of {500, 100, 20, 1} takes around 0.62 mb just to store the weights, bias, and 1st and 2nd moments. Therefore, all variables must be allocated to the heap using malloc().

The main struct for the network is also created, and default values are assigned to all variables so that the network will run without the user setting every variable's value.

## He Initialization

All weights and biases are initialized with small, random values, typically between \-0.05 and 0.05 in order to prevent dying neurons, stop each weight from changing by the same amount, and prevent symmetrical networks. If every neuron in a layer has the same initial value, they will all compute the same output and receive the same weight updates, preventing the network from learning complex data. For this reason, all weights and biases in a network must be set to random values.

The network uses He weight and bias initialization in order to prevent exploding or dying gradients when using the ReLU or Leaky ReLU activation functions.

He initialization scales weights by a factor of sqrt(2 / previousLayerNeurons)

This means that output layer neurons will have much larger starting values that neurons closer to the input layer. 

<details>
<summary>Click to view Code</summary>

```c
//Initialize Weights and Bias with scaled random values (He Initialization)
    for (int i = 0; i < net->layers; i++) {

        // Inputs size loops for first layer, previous layer size for other layers
        loops = (i == 0) ? data->inputSize : neuronLayers[i - 1];

        // Scale values based on size of previous layer - sqrt(2 / previous layer size)
        scale = sqrt(2.0f / loops);

        // For each neuron in the layer
        for (int j = 0; j < neuronLayers[i]; j++) {

            // Set Bias to 0
            net->B[i][j] = 0;

            // Set each weight to scaled random value
            for (int k = 0; k < loops; k++) {
                net->W[i][j][k] = (((float)rand() / RAND_MAX) * 2 - 1) * scale;
            }
        }
    }  
```

</details>

## Data Normalization

All data and labels must be normalized in order for the inputs to have equal importance and to prevent large numbers during backpropagation.

To see why this is needed, consider there are 2 inputs: house cost and number of bedrooms. House cost ranges from 100,000 to 1,000,000 while number of bedrooms ranges from 1 to 5\.

If y \= (w1 x housecost) \+ (w2 x number of bedrooms) \+ b, and w1 and w2 both start with values ranging from \-0.05 to 0.05, then naturally housecost will contribute to almost the entire output. Even as the network learns and adjusts weights, it will be difficult to scale w2 significantly while keeping w1 very small.

Not only this, but during backpropagation error is traced back by multiplying the error and weights. As error is traced back layer to layer, this number will get very large and eventually escape the range of an int or float. 

For these reasons, data is normalized between 0 and 1\. All inputs have equal importance at the start of training, and backpropagation is able to continue for many layers without extremely large values.

<details>  
<summary>Click to view code</summary>

```c

// ------ Normalize Data ------
    //Set max values to 0
    for (int i = 0; i < data->inputSize + 1; i++) net->maxValues[i] = 0;

    //Loop through data to find max of each input and output
    for (int i = 0; i < data->dataSize; i++) {

        // Find max of inputs
        for (int j = 0; j < data->inputSize; j++) {
            if (data->x[i][j] > net->maxValues[j])
                net->maxValues[j] = data->x[i][j];
        }

        // Find max of outputs
        if (data->y[i] > net->maxValues[data->inputSize])
            net->maxValues[data->inputSize] = data->y[i];
    }

    // Loop through all data and divide by max
    for (int i = 0; i < data->dataSize; i++) {

        // Divide inputs by max
        for (int j = 0; j < data->inputSize; j++)
            data->x[i][j] /= net->maxValues[j];

        // Divide output by max
        data->y[i] /= net->maxValues[data->inputSize];
    }

```  
</details>

## Forward Pass

The forward pass is how the network takes inputs and produces an output. In a neural network, this is done by starting at the inputs, calculating the values for the first layer of neurons, and then using the previous layer of neurons as inputs for the next layer. The steps for this are shown below:

1. For each neuron, inputs are multiplied by weights and a bias is added, similar to in perceptron.c

   Z \= ∑wx \+ b

2. Next, Z is passed through an activation function f(x). This is done to allow the model to generalize to data that is not linearly separable

   A \= f(Z)

   

3. At this point, each neuron in the first layer has computed an output, A. Now, the next layer performs the same sequence of calculations, but by using A as the input. This process repeats for each hidden layer in the network.

   Z \= ∑wA \+ b

   A \= f(Z)

   

4. For the final output, no activation function is applied.\* This is because activation functions typically limit what the value can be. For example, in ReLU there can be no values under 0 and in the sigmoid function all values must be between 0 and 1\. This is needed for training, but should not be used on the output

   \*In the library, an activation function is technically applied to the output neuron for the purpose of making the loops more easy to read, but the output is just taken from the preactivated value, Z. 

   \*For classification problems, where there is an entire layer of output neurons, a softmax function is typically applied to the output neurons so that for the neurons in the final output layer, the sums add up to 1\. 

<details>
<summary>Click to view code</summary>

```c

// --- Forward Pass ---
            //For each layer
            for (int j = 0; j < net->layers; j++) {

                //For each neuron
                for (int k = 0; k < net->neuronLayers[j]; k++) {

                    //Dropout - Randomly drop neurons during training to prevent overfitting
                    if (net->dropout && j < net->layers - 1) {

                        //Random chance to drop neuron
                        if (((float)rand() / RAND_MAX) < net->dropoutChance) {

                            //Set value to 0
                            net->A[j][k] = 0;

                            //Exit loop
                            continue;
                        }
                    }

                    //Set neuron value to bias
                    net->Z[j][k] = net->B[j][k];

                    //If first layer, add dot product of all inputs and weights
                    if (j == 0) for (int z = 0; z < data->inputSize; z++) net->Z[j][k] += data->x[i][z] * net->W[j][k][z];

                    //If not first layer, add dot product of all activations from previous layer and weights
                    else for (int z = 0; z < net->neuronLayers[j - 1]; z++) net->Z[j][k] += net->A[j - 1][z] * net->W[j][k][z];

                    //Activation Function
                    switch(net->activationFunction) {

                        //ReLU
                        case 'R':
                            net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0;
                            break;

                        //Leaky ReLU
                        case 'L':
                            net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0.01f * net->Z[j][k];
                            break;

                        //Sigmoid
                        case 'S':
                            net->A[j][k] = 1 / (1 + exp(-net->Z[j][k]));
                            break;
                        
                        //None
                        default:
                            net->A[j][k] = net->Z[j][k];
                            break;
                    }
                }
            }
``` 
</details>

## Backpropagation

### Assigning Blame

In order to adjust the parameters of each neuron, it is necessary to find how much each neuron contributed to the total error. This is done by starting with the total error of the system, and working back layer by layer to find how much each neuron contributed to the total error. The steps are shown below:

1. The output neuron is directly responsible for the error, so its blame is equal to the total error  
   Blame \= eTotal

2. The neurons in the previous layer are responsible for as much of the total error as they contributed to the output neuron. Multiply the blame of the neuron in the next layer by the weight connecting the current neuron and that output neuron. If the current neuron is connected to many neurons, add the result of each multiplication   
   Blame \= 0   
   For every neuron in next layer:   
   Blame \+= previousNeuronBlame \* connectingWeight

3. Multiply the total balance for that neuron by the derivative of the activation function. Most activation functions for neural networks are specifically designed to have derivatives that are easy to calculate.

<details>
<summary>Click to view code</summary>

```c
 
// --- Backpropagation ---

            //Step 1: Calculate Blame for every neuron, starting with output layer

            //For output layer, delta is total error
            net->D[net->layers - 1][0] = eTotal;

            //For each layer
            for (int j = net->layers - 2; j >= 0; j--) {

                //For each neuron
                for (int k = 0; k < net->neuronLayers[j]; k++) {

                    //Set Delta to 0
                    net->D[j][k] = 0.0f;

                    //Sum of Deltas from layer ahead * corresponding weights
                    for (int z = 0; z < net->neuronLayers[j + 1]; z++) {
                        net->D[j][k] += net->D[j + 1][z] * net->W[j + 1][z][k];
                    }

                    //Activation Function Derivative
                    switch (net->activationFunction) {

                        //ReLU Derivative
                        case ('R'):
                            net->D[j][k] *= (net->Z[j][k] > 0) ? 1.0f : 0.0f;
                            break;
                        
                        //Leaky ReLU Derivative
                        case ('L'):
                            net->D[j][k] *= (net->Z[j][k] > 0) ? 1.0f : 0.01f;
                            break;

                        //Sigmoid Derivative
                        case ('S'):
                            net->D[j][k] *= net->A[j][k] * (1 - net->A[j][k]);

                            //Needs gradient clipping
                            if (net->D[j][k] > net->clip) net->D[j][k] = net->clip;
                            if (net->D[j][k] < -net->clip) net->D[j][k] = -net->clip;

                            break;
                        
                        //If no activation function, derivative = 1;
                    }
                }
            }
``` 
</details>

### Weight Updates

Now that each neuron has its blame assigned for how much it contributed to the final error, it is possible to update the weights and bias for each neuron. This is also where optimizers are used. Optimizers alter the weight and bias calculation to help the network learn more quickly.

1. Update bias based off of learningRate and Blame   
   Bias \+= LEARNING\_RATE \* Blame  
     
2. Update weights based off of learningRate, Blame, and the previous layers output   
   Weight \+= LEARNING\_RATE \* Blame \* PreviousLayerOutput

<details>
<summary>Click to view code</summary>

```c
//Step 2: Update Weights and Biases using Deltas
            //For each layer
            for (int j = 0; j < net->layers; j++) {

                //For each neuron
                for (int k = 0; k < net->neuronLayers[j]; k++) {

                    weightLength = 0;

                    //Update Bias
                    switch (net->optimizer) {

                        //Adam Optimizer
                        case 'A':

                            //Calculate 1st and 2nd moments
                            net->VelocityB[j][k] = net->momentumDecay * net->VelocityB[j][k] + (1 - net->momentumDecay) * net->D[j][k];
                            net->ScalingB[j][k] = net->scalingDecay * net->ScalingB[j][k] + (1 - net->scalingDecay) * net->D[j][k] * net->D[j][k];

                            //Bias Correction
                            correctedA = net->VelocityB[j][k] / (1 - pow(net->momentumDecay, epoch));
                            correctedB = net->ScalingB[j][k] / (1 - pow(net->scalingDecay, epoch));

                            //Update Bias
                            net->B[j][k] += net->LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                            break;

                        //RMSProp Optimizer
                        case 'R':
                            //Calculate 2nd moment
                            net->ScalingB[j][k] = net->scalingDecay * net->ScalingB[j][k] + (1 - net->scalingDecay) * net->D[j][k] * net->D[j][k];

                            //Update Bias
                            net->B[j][k] += net->LEARNING_RATE / (sqrt(net->ScalingB[j][k]) + 1e-8) * net->D[j][k];
                            break;

                        //Momentum
                        case 'M':

                            //Calculate Momentum
                            net->VelocityB[j][k] = net->momentumDecay * net->VelocityB[j][k] + (1 - net->momentumDecay) * net->D[j][k];
                            net->B[j][k] += net->VelocityB[j][k] * net->LEARNING_RATE;
                            break;
                        
                        //No Optimizer
                        default:
                            net->B[j][k] += net->LEARNING_RATE * net->D[j][k];
                            break;
                    }
                    
                    //Loops needed to update all weights
                    loops = (j == 0) ? data->inputSize : net->neuronLayers[j - 1];

                    //For each weight
                    for (int z = 0; z < loops; z++) {
                        
                        //Calculate current gradient
                        currentGradient = (j == 0) ? net->D[j][k] * data->x[i][z] : net->D[j][k] * net->A[j - 1][z];

                        //Update Weights
                        switch (net->optimizer) {

                            //Adam Optimizer
                            case 'A':

                                //Calculate 1st and 2nd moments
                                net->Velocity[j][k][z] = net->momentumDecay * net->Velocity[j][k][z] + (1 - net->momentumDecay) * currentGradient;
                                net->Scaling[j][k][z] = net->scalingDecay * net->Scaling[j][k][z] + (1 - net->scalingDecay) * currentGradient * currentGradient;

                                //Bias Correction for Adam
                                correctedA = net->Velocity[j][k][z] / (1 - pow(net->momentumDecay, data->trainingSize * epoch));
                                correctedB = net->Scaling[j][k][z] / (1 - pow(net->scalingDecay, data->trainingSize * epoch));

                                //Update Weight with Adam
                                net->W[j][k][z] += net->LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                                break;

                            //RMSProp Optimizer
                            case 'R':
                                //Calculate 2nd moment
                                net->Scaling[j][k][z] = net->scalingDecay * net->Scaling[j][k][z] + (1 - net->scalingDecay) * currentGradient * currentGradient;

                                //Update Weights
                                net->W[j][k][z] += net->LEARNING_RATE / (sqrt(net->Scaling[j][k][z]) + 1e-8) * currentGradient;
                                break;

                            //Momentum
                            case 'M':

                                //Calculate Momentum
                                net->Velocity[j][k][z] = net->momentumDecay * net->Velocity[j][k][z] + (1 - net->momentumDecay) * currentGradient;

                                //Update Weights
                                net->W[j][k][z] += net->Velocity[j][k][z] * net->LEARNING_RATE;
                                break;

                            //No Optimizer
                            default:
                                net->W[j][k][z] += net->LEARNING_RATE * currentGradient;
                                break;
                        }

                        //Max Norm Regulation - Limit the maximum norm of the weights to prevent exploding gradients
                        if (net->maxNormRegulation) weightLength += net->W[j][k][z] * net->W[j][k][z];
                    }

                    if (net->maxNormRegulation) {
                        weightLength = sqrt(weightLength);
                        if (weightLength > net->maxNorm) {
                            for (int z = 0; z < loops; z++) {
                                net->W[j][k][z] = net->maxNorm;
                            }
                        }
                    }
                }
            }
        }
```
</details>

## Testing

After the model is done being trained, it is tested on unseen data. Since the network has not been trained on the data, testing error is almost always higher than training error.

Testing works by running new data through the forward pass, and calculating the average error. No weight adjustments or backpropagation are typically done during testing.

One common problem with neural networks is overfitting, where the model is able to “memorize” training data and have almost no error while training, but has high error during testing.

<details>
<summary>Click to view code</summary>

```c

void testNetwork(Network*net, dataSet *data) {

    float eTestingAvg = 0;

    //Loop through testing data
    for (int i = data->trainingSize; i < data->dataSize; i++) {

        // For each layer
        for (int j = 0; j < net->layers; j++) {

            // For each neuron
            for (int k = 0; k < net->neuronLayers[j]; k++) {

                //Set neuron value to bias
                net->Z[j][k] = net->B[j][k];

                //If first layer, use inputs
                if (j == 0) for (int z = 0; z < data->inputSize; z++) net->Z[j][k] += data->x[i][z] * net->W[j][k][z];

                //Else, use outputs from previous layer
                else for (int z = 0; z < net->neuronLayers[j - 1]; z++) net->Z[j][k] += net->A[j - 1][z] * net->W[j][k][z];

                //Activation Function
                switch(net->activationFunction) {

                    //ReLU
                    case 'R':
                        net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0;
                        break;

                    //Leaky ReLU
                    case 'L':
                        net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0.01f * net->Z[j][k];
                        break;

                    //Sigmoid
                    case 'S':
                        net->A[j][k] = 1 / (1 + exp(-net->Z[j][k]));
                        break;
                    
                    //None
                    default:
                        net->A[j][k] = net->Z[j][k];
                        break;
                }
            }
        }

        //Calculate Abs Average Error
        eTestingAvg += fabs((data->y[i] - net->Z[net->layers - 1][0]) / data->y[i]);
    }

    //Calculate Average Error for Epoch
    eTestingAvg = (eTestingAvg / data->testingSize) * 100;

    printf("Testing Average Error: %.2f\n", eTestingAvg);
}
```
</details>

# Hyperparameters

Hyperparameters are the user-adjusted variables that can control how the network learns and what features it uses.

## Activation Function

![Common Activation Functions](Perceptron_Project/images/activationFunctions.png)

An activation function is what allows a network to generalize to non-linear data. An activation function is typically represented as `A = f(z)` where z is the preactivation value. Activation functions must be computationally light, both for the forward pass and while finding the derivative for backpropagation. This section will cover some of the most common activation functions.

### ReLU

ReLU is the most common activation function and is used in most large language models today. ReLU takes in the input, and if it is under zero it is set to zero. 

Code: `net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0;`

Derivative: `net->D[j][k] *= (net->Z[j][k] > 0) ? 1.0f : 0.0f;`  
   
Pros:

- Very easy to calculate both on the forward and backward pass  
- Prevents vanishing gradients while Z \> 0

Cons:

- Dying ReLU: Neurons will often “die” and become stuck outputting zero. If a neuron outputs zero at any given time, its blame becomes 0, preventing any weight updates and keeping it stuck at zero. This results in many neurons becoming inactive after training.

### Leaky ReLU

Leaky ReLU was created to solve the main problem with ReLU: dying neurons. Instead of keeping neurons stuck at 0 once they become negative, Leaky ReLU multiplies their value by 0.01, allowing them to stay alive and potentially become active again.

Code: `net->A[j][k] = (net->Z[j][k] > 0) ? net->Z[j][k] : 0.01f;`

Derivative: `net->D[j][k] *= (net->Z[j][k] > 0) ? 1.0f : 0.01f;`  
   
Pros:

- Very easy to calculate both on the forward and backward pass  
- Prevents vanishing gradients 

Cons:

- Slightly more computations than for ReLU, partly because more neurons are active and need to have weight updates

### Sigmoid

The sigmoid activation function is typically used for binary classification problems since the output of neurons is constrained between 0 and 1\. This activation function is not typically used anymore due to the high computational complexity and vanishing gradients.

Code: `net->A[j][k] = 1 / (1 + exp(-net->Z[j][k]));`

Derivative: `net->D[j][k] *= net->A[j][k] * (1 - net->A[j][k]);`  
   
Pros:

- Always outputs between a 0 and 1  
- Smooth gradients across all z values

Cons:

- Computationally expensive calculating exponential value  
- Vanishing gradients at high or low values

## Optimizers

Optimizers are algorithms designed to speed up the training process. The optimizers used in the library are momentum, RMSProp, or Adam.

### Momentum

Momentum is designed to speed up learning by adding another term to weight updates that keeps them moving in the same direction, giving them “momentum.”   
\*\*\*In this library, a modified calculation of momentum is used to match the calculation used for Adam

Code:   
```c

 //Calculate Momentum
net->Velocity[j][k][z] = net->momentumDecay * net->Velocity[j][k][z] + (1 - net->momentumDecay) * currentGradient;

//Update Weights
net->W[j][k][z] += net->Velocity[j][k][z] * net->LEARNING_RATE;

 
```

### RMSProp

RMSProp is an algorithm that varies the learning rate for each weight based on the current gradient of the weight.

Code: 

```c

//Calculate 2nd moment
net->Scaling[j][k][z] = net->scalingDecay * net->Scaling[j][k][z] + (1 - net->scalingDecay) * currentGradient * currentGradient;

//Update Weights
net->W[j][k][z] += net->LEARNING_RATE / (sqrt(net->Scaling[j][k][z]) + 1e-8) * currentGradient;

```

### Adam

Adam is the most common optimizer and is used in almost every large neural network. Adam is a combination of momentum and RMSProp.

Code:

```c

//Calculate 1st and 2nd moments
net->Velocity[j][k][z] = net->momentumDecay * net->Velocity[j][k][z] + (1 - net->momentumDecay) * currentGradient;
net->Scaling[j][k][z] = net->scalingDecay * net->Scaling[j][k][z] + (1 - net->scalingDecay) * currentGradient * currentGradient;

//Bias Correction for Adam
correctedA = net->Velocity[j][k][z] / (1 - pow(net->momentumDecay, data->trainingSize * epoch));
correctedB = net->Scaling[j][k][z] / (1 - pow(net->scalingDecay, data->trainingSize * epoch));

//Update Weight with Adam
net->W[j][k][z] += net->LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;


```

## Features

Features are extra functions or steps that can allow a model to learn faster, generalize better, or train more consistently.

### Dropout

 Dropout randomly sets neurons to 0 during the forward pass of training resulting in them not contributing to the output, and scales the outputs of the remaining neurons to make up for the lost neurons. This is used to prevent networks from overfitting to training data. Dropout significantly increases the time required to train the network due to the different combinations of neurons that need to be trained. 

### Max Norm Regularization: 

Max Norm regularization is a way to control the impact of a single neuron on the final output. The length of the weight vector is taken by taking the square root of every weight in a neuron squared and added together. This is done to prevent exploding gradients and ensure that more neurons are able to stay active.  
Length Formula: length \= sqrt(w1^2 \+ w2^2 \+ … \+ wn^2)

### Early Stopping

Early stopping is a way for training to automatically stop if error is increasing or stop decreasing. This is more useful in the single neuron network perceptron.c where error should steadily decrease as opposed to large neural networks where error may temporarily increase to escape local minima.

## Hyperparameters

**Learning Rate:** How fast the network adjusts parameters based on error. Too high of a learning rate results in jumping around, exploding gradients, and being unable to settle in the minima while too low of a learning rate results in high training time and getting stuck in poor local minima.

**Epochs**: How many times to run through the data during training. More epochs results in more training and a lower error for training values, but overtraining can lead to overfitting and higher testing error. Epochs are directly related to how much time it takes to train.

**momentumDecay:** Used for momentum and Adam optimizer. The typical value is 0.9. Higher values result in more momentum and weights being carried over from past updates, while a lower value results in less updates based on past weight changes.

**scalingDecay**: Used for RMSProp and Adam Optimizer. The typical value is 0.9 to 0.999. This controls how much of past squared gradients are retained and applied to weight updates.

**maxNorm**: Max norm controls the maximum weight vector length for a neuron, or how large the weights can get as a group. A higher value allows individual neurons to have a large impact, but can lead to exploding gradients. A lower value can cap how much neurons contribute to the output, but too low of a value will result in many neurons hitting the limit and unable to adjust weights as needed.

**dropoutChance**: Dropout rate controls the chance of a neuron dropping out during the forward pass during training if using the dropout feature. Values range from 0 to 1 (0% to 100%). A typical value is 0.1 to 0.5. A higher value results in more neurons being dropped, and a much larger training time due to more combinations of neurons needing to be trained. 

**Print Interval**: How often to print messages during training. For example, a print interval of 100 means results are printed every 100 epochs.

**Min Stopping Epoch**: How early the network is able to stop during the earlyStopping feature. This prevents the network from stopping training immediately after starting.

**dataSize**: How many samples of data or rows are in the dataset

**inputSize**: How many different features or columns are in the dataset

**trainingSize:** How much of dataSize to use for training. Typically around 70%. The rest of the data is reserved for testing.

## Architecture Hyperparameters

In this library, there is one array to define the entire network architecture. neuronLayers defines how many neurons are in each of the layers, and the variable “layers“ is later defined based on how long the array neuronLayers is.

Example:  
int neuronLayers\[\] \= {100, 50, 20, 1};  
int layers \= 4;

This will create a network with a first hidden layer with 100 neurons, connected to a layer with 50 neurons, connected to a layer with 20 neurons, to a final output layer with 1 neuron.

Inputs \-\> 100 neuron layer \-\> 50 neuron layer \-\> 20 neuron layer \-\> final output neuron

Rules:

- Layers must match the length of neuronLayers  
- The final layer in neuronLayers must have 1 neuron  
- When storing and loading weights, the network must have the same architecture when loading weights as it did when the weights were stored.  
  - Ex: Saved weights after training with neuronLayers \= {50, 20, 1}. Things like learning rate, epochs, and most hyperparameters can be changed but neuronLayers must remain the same when the weights are loaded at a future time. For best performance, the activation function should also be kept the same as the weights will not be accurate across different activation functions.

## Functions

### createDataSet

Parameters: (float \*)x, y, DATA\_SIZE, INPUT\_SIZE, TRAINING\_SIZE);  
Returns: data struct

This function stores and formats all data into arrays and stores them in a struct to be more easily passed later. This function must be called directly after the data is pasted in and the data hyperparameters have been defined, but before any other functions.

### createNetwork

Parameters: neuronLayers, data, layers  
Returns: net struct

This function initializes all memory for the network, normalizes all input data, and initializes all weights and biases with small random values using HE Initialization. This must be called after the createDataSet function and after neuronLayers and layers have been defined. This will return net, which stores every variable for the network besides data.

### trainNetwork

Parameters: net, data  
Returns: \-

This function trains the network and updates all variables in place. This can be called anytime after the network and data have been defined.

### testNetwork

Parameters: net, data  
Returns: \-

This function tests the network on the data not used for training. It will print testing accuracy at the end of the function. This function can be called anytime after the network and data have been defined.

### predictOutput

Parameters: net, test, data  
Returns: prediction

This function takes in new, user defined data, test. Test is an array of length inputSize that can have any inputs the user wants to pass through the neural network. The function will return the predicted value. This can be called anytime after the network and data have been defined.

### loadWeights

Parameters: net, “networkWeights.bin”  
Returns: 1 or 0 for success or failure

This function loads weights and biases from the file “networkWeights.bin” if they have been saved. This allows the network to resume training or testing where it left off instead of starting from scratch.

When storing and loading weights, the network must have the same architecture when loading weights as it did when the weights were stored.

- Ex: Saved weights after training with neuronLayers \= {50, 20, 1}. Things like learning rate, epochs, and most hyperparameters can be changed but neuronLayers must remain the same when the weights are loaded at a future time. For best performance, the activation function should also be kept the same as the weights will not be accurate across different activation functions.

### saveWeights

Parameters: net, “networkWeights.bin”  
Returns: 1 or 0 for success or failure

This function saves the current weights and biases for the network into the file “networkWeights.bin”. This can be used after training to allow the optimized values to be loaded later. The file “networkWeights.bin” is reset every time this function is called.

### freeMemory

Parameters: net  
Returns: \- 

This function frees all allocated memory from the struct net. This should be called at the end of the file.

### freeDataSet

Parameters: data  
Returns: \-

This function frees all allocated memory from the struct data. This should be called at the end of the file.  
