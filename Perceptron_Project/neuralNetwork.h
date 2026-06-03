#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

//Network Struct
typedef struct {

    //Parameters
    float ***W;
    float **B;

    //Adam Optimizer Parameters
    float ***Velocity;
    float **VelocityB;
    float ***Scaling;
    float **ScalingB;

    //Values
    float **Z;
    float **A;
    float **D;
    float *maxValues;
    int layers;
    int *neuronLayers;

    //Training
    int EPOCHS;                  //Amount of Times to Go Through Entire Dataset
    float LEARNING_RATE;            //How Fast Weights change based on Error
    int PRINT_INTERVAL;           //How Often to Print Results (in Epochs)
    int MIN_STOPPING_EPOCH;        //Minimum Epochs before Early Stopping can Occur
    int inputSize;

    //Feature Hyperparameters
    float dropoutChance;           //Chance to drop each neuron during training - 0 is 0%, 1 is 100% change of dropping
    float maxNorm;                  //Maximum norm for weights if maxNormRegulation is enabled
    float momentumDecay;           //Momentum factor
    float scalingDecay;          //Scaling factor for learning rate decay
    float clip;                    //Value to clip gradients for sigmoid activation function to prevent exploding gradients

    //Features
    bool earlyStopping;          //Whether to Stop Training if Error stops decreasing
    bool dropout;                //Whether to randomly drop neurons during training to prevent overfitting
    bool maxNormRegulation;      //Whether to cap weights to prevent exploding gradients and overfitting

    //Optimizer
    char optimizer;
    //A = Adam Optimizer | Optimal Learning Rate = 0.0003
    //R = RMSProp        | Optimal Learning Rate = 0.00005
    //M = Momentum       | Optimal Learning Rate = 0.5
    //N = None           | Optimal Learning Rate = 0.2

    //Activation Function
    char activationFunction;
    //L = Leaky ReLU
    //R = ReLU
    //S = Sigmoid

} Network;

//Struct for user data
typedef struct {
    float **x;
    float  *y;
    int dataSize;
    int inputSize;
    int trainingSize;
    int testingSize;
} dataSet;

//Function Prototypes
dataSet* createDataSet(float *xFlat, float *y, int dataSize, int inputSize, int trainingSize);
void freeDataSet(dataSet *data);

void testNetwork(Network *net, dataSet *data);
void freeMemory(Network *net);
float predictOutput(Network *net, float *Inputs, dataSet *data);
Network* createNetwork(int *neuronLayers, dataSet *data, int layers);
void trainNetwork(Network *net, dataSet *data);

int saveWeights(Network *net, const char *filename);
int loadWeights(Network *net, const char *filename);


#endif