#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

//Data
#define DATA_SIZE 20                //Amount of Data Points
#define INPUT_SIZE 3                //Number of Different Inputs / Parameters
#define TRAINING_SIZE 15            //How many data points to use for training

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

int main() {

    //INPUTS: Sq footage, bedrooms, yard size
    float x[DATA_SIZE][INPUT_SIZE] = {
        {850, 1, 500},
        {1200, 2, 1000},
        {950, 2, 750},
        {1800, 3, 1500},
        {2200, 4, 2000},
        {1500, 3, 1200},
        {3000, 5, 3000},
        {1100, 2, 800},
        {2600, 4, 2500},
        {700, 1, 400},
        {1750, 3, 1300},
        {2900, 4, 2800},
        {1350, 2, 900},
        {2100, 3, 1600},
        {500, 1, 600},
        {1650, 3, 1400},
        {2400, 4, 2200},
        {1050, 2, 850},
        {3200, 5, 3500},
        {1900, 3, 1800}};

    //Ex. Result Price ($)
    //Linear Labels
    //float y[] = {120000, 185000, 140000, 280000, 350000, 230000, 500000, 160000, 420000, 95000, 270000, 470000, 200000, 330000, 75000, 255000, 390000, 155000, 540000, 300000};

    //Non-Linear Labels
    float y[] = {95000, 210000, 125000, 480000, 890000, 370000, 2100000, 175000, 1400000, 72000, 460000, 1850000, 240000, 750000, 52000, 420000, 1150000, 162000, 2800000, 580000};

    //Architecture
    int neuronLayers[] = {50, 20, 1};    //Array of Neuron Counts for Each Layer

    int layers = 3;

    //Create Data
    dataSet *data = createDataSet((float *)x, y, DATA_SIZE, INPUT_SIZE, TRAINING_SIZE);

    //Create all variables for network
    Network *net = creatNetwork(neuronLayers, data, layers);

    //Variables -- Declare any neededed
    net->EPOCHS = 1000;                  //Amount of Times to Go Through Entire Dataset
    net->LEARNING_RATE = 0.2;            //How Fast Weights change based on Error
    net->PRINT_INTERVAL = 100;           //How Often to Print Results (in Epochs)

    //Optimizer
    net->optimizer = 'N';
    //A = Adam Optimizer | Optimal Learning Rate = 0.0003
    //R = RMSProp        | Optimal Learning Rate = 0.00005
    //M = Momentum       | Optimal Learning Rate = 0.5
    //N = None           | Optimal Learning Rate = 0.2

    //Activation Function
    net->activationFunction = 'L';
    //L = Leaky ReLU
    //R = ReLU
    //S = Sigmoid

    //Train Network
    trainNetwork(net, data);

    //Test Network
    testNetwork(net, data);

    //Try Predicting an Output
    float test[] = {0.5, 0.5, 0.5};
    float prediction = predictOutput(net, test, data);

    //Free memory when program is done
    freeMemory(net);

    freeDataSet(data);

    return 0;
}

dataSet* createDataSet(float *xFlat, float *y, int dataSize, int inputSize, int trainingSize) {
    dataSet *data = malloc(sizeof(dataSet));
    data->dataSize    = dataSize;
    data->inputSize   = inputSize;
    data->trainingSize = trainingSize;
    data->testingSize  = dataSize - trainingSize;

    // Copy labels
    data->y = malloc(sizeof(float) * dataSize);
    memcpy(data->y, y, sizeof(float) * dataSize);

    // Build float** from the flat row-major layout of the user's 2D array
    data->x = malloc(sizeof(float *) * dataSize);
    for (int i = 0; i < dataSize; i++) {
        data->x[i] = malloc(sizeof(float) * inputSize);
        memcpy(data->x[i], xFlat + i * inputSize, sizeof(float) * inputSize);
    }

    return data;
}

Network* createNetwork(int *neuronLayers, dataSet *data, int layers) {
    Network* net = malloc(sizeof(Network));

    //Find Amount of Layers
    net->layers = layers;

    //Architecture
    net->neuronLayers = neuronLayers;

    //Max Values for Normalization
    net->maxValues = malloc(sizeof(float) * (data->inputSize + 1));

    // ---------- Loop to declare all arrays needed to heap ----------
    //Weights
    net->W = malloc(sizeof(float **) * net->layers);

    //1st Moment / Velocity for each Weight
    net->Velocity = malloc(sizeof(float **) * net->layers);

    //1st Moment / Velocity for each Bias
    net->VelocityB = malloc(sizeof(float *) * net->layers);

    //2nd Moment / Scaling Factor for each Weight (for Adam Optimizer)
    net->Scaling = malloc(sizeof(float **) * net->layers);

    //2nd Moment / Scaling Factor for each Bias (for Adam Optimizer)
    net->ScalingB = malloc(sizeof(float *) * net->layers);

    //Bias
    net->B = malloc(sizeof(float *) * net->layers);

    //Preactiviation Value
    net->Z = malloc(sizeof(float *) * net->layers);

    //Activation Value
    net->A = malloc(sizeof(float *) * net->layers);

    //Delta for Backpropagation
    net->D = malloc(sizeof(float *) * net->layers);

    for (int i = 0; i < net->layers; i++) {

        //Allocate Memory for Each Layer
        net->W[i] = malloc(sizeof(float *) * neuronLayers[i]);
        net->Velocity[i] = malloc(sizeof(float *) * neuronLayers[i]);
        net->VelocityB[i] = malloc(sizeof(float) * neuronLayers[i]);
        net->Scaling[i] = malloc(sizeof(float *) * neuronLayers[i]);
        net->ScalingB[i] = malloc(sizeof(float) * neuronLayers[i]);
        net->B[i] = malloc(sizeof(float) * neuronLayers[i]);
        net->Z[i] = malloc(sizeof(float) * neuronLayers[i]);
        net->A[i] = malloc(sizeof(float) * neuronLayers[i]);
        net->D[i] = malloc(sizeof(float) * neuronLayers[i]);

        //Create 3d array for Weights and Velocity
        for (int j = 0; j < neuronLayers[i]; j++) {
            net->W[i][j] = malloc(sizeof(float) * ((i == 0) ? data->inputSize : neuronLayers[i - 1]));
            net->Velocity[i][j] = malloc(sizeof(float) * ((i == 0) ? data->inputSize : neuronLayers[i - 1]));
            net->Scaling[i][j] = malloc(sizeof(float) * ((i == 0) ? data->inputSize : neuronLayers[i - 1]));

            //Set all Velocity and Scaling to 0
            net->ScalingB[i][j] = 0;
            net->VelocityB[i][j] = 0;
            for (int k = 0; k < ((i == 0) ? data->inputSize : neuronLayers[i - 1]); k++) {
                net->Velocity[i][j][k] = 0;
                net->Scaling[i][j][k] = 0;
            }
        }
    }

    // ------Default Hyperparameters and Features------

    //Training
    net->EPOCHS = 1000;                  //Amount of Times to Go Through Entire Dataset
    net->LEARNING_RATE = 0.3;            //How Fast Weights change based on Error
    net->PRINT_INTERVAL = 100;           //How Often to Print Results (in Epochs)
    net->MIN_STOPPING_EPOCH = 50;        //Minimum Epochs before Early Stopping can Occur
    net->dropoutChance = 0.5;            //Chance to drop each neuron during training - 0 is 0%, 1 is 100% change of dropping
    net->maxNorm = 1.5;                  //Maximum norm for weights if maxNormRegulation is enabled
    net->momentumDecay = 0.90;           //Momentum factor
    net->scalingDecay = 0.990;           //Scaling factor for learning rate decay
    net->clip = 0.5f;                    //Value to clip gradients for sigmoid activation function to prevent exploding gradients

    //Features
    net->earlyStopping = false;          //Whether to Stop Training if Error stops decreasing
    net->dropout = false;                //Whether to randomly drop neurons during training to prevent overfitting
    net->maxNormRegulation = true;       //Whether to cap weights to prevent exploding gradients and overfitting

    //Optimizer and Activation function
    net->optimizer = 'N';
    net->activationFunction = 'L';

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
    printf("Data Normalized\n");

    // ------ Initialize Weights and Biases ------
    // Seed Random Number Generator
    srand(time(NULL));

    int loops = 0;
    float scale = 0;

    // Initialize Weights and Bias with scaled random values (He Initialization)
    for (int i = 0; i < net->layers; i++) {

        // Inputs size loops for first layer, previous layer size for other layesr
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

    //Final Print
    printf("Weights and Biases Allocated and Randomly Initialized\n");

    return net;
}

void trainNetwork(Network *net, dataSet *data) {

    //All varibles needed later
    float eTotal = 0, eTrainingAvg = 0, lastEAvg = 1000, hiddenError = 0, scale = 0, currentGradient = 0, loops = 0, correctedA, correctedB = 0, weightLength = 0;

    //START TIMING
    clock_t start = clock();

    //Network Training Loop
    for (int epoch = 1; epoch <= net->EPOCHS; epoch++) {

        //Reset Average Error for each Epoch
        eTrainingAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < data->trainingSize; i++) {

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

            //Calculate Total Error: Label - Preactivation Output Neuron Value
            eTotal = data->y[i] - net->Z[net->layers - 1][0];

            //Calculate Average Error for Prints
            eTrainingAvg += fabs(eTotal / data->y[i]);

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

        //Calculate Average Error for Epoch
        eTrainingAvg = eTrainingAvg / data->trainingSize * 100;

        //Mark continuous timing and calculate total runtime
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % net->PRINT_INTERVAL == 0) {

            //Print Epoch, Average Error, Runtime, and Result
            printf("Epoch: %i | Average Error: %.4f | Runtime: %.1f | Result: %.3f\n", epoch, eTrainingAvg, runtime * 1000, net->Z[net->layers - 1][0]);

            //for (int j = 0; j < net->layers; j++) {
            //   for (int k = 0; k < neuronLayers[j]; k++) {
            //        printf("D[%d][%d] = %.6f | W[%d][%d][0] = %.6f\n", j, k, net->D[j][k], j, k, net->W[j][k][0]);
            //    }
            //}
        }

        //Stop early if error improvement is less than 0.001 and using early stopping
        if (net->earlyStopping && lastEAvg - eTrainingAvg < 0.001 && epoch > net->MIN_STOPPING_EPOCH) {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eTrainingAvg);
            break;
        }

        //Update lastEAvg for Early Stopping Check
        lastEAvg = eTrainingAvg;
    }
}

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

float predictOutput(Network *net, float *Inputs, dataSet *data) {
 
    //For each layer
    for (int j = 0; j < net->layers; j++) {

        //For each neuron
        for (int k = 0; k < net->neuronLayers[j]; k++) {

            //Set neuron value to bias
            net->Z[j][k] = net->B[j][k];

            //If first layer, use inputs
            if (j == 0) for (int z = 0; z < data->inputSize; z++) net->Z[j][k] += Inputs[z] * net->W[j][k][z];

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

    //Print predicted value
    printf("Predicted Value: %.3f\n", net->Z[net->layers - 1][0]);

    //Return Final Output
    return net->Z[net->layers - 1][0];
}

void freeDataSet(dataSet *data) {
    for (int i = 0; i < data->dataSize; i++)
        free(data->x[i]);
    free(data->x);
    free(data->y);
    free(data);
}

void freeMemory(Network *net) {

    //Free Allocated Memory
    for (int i = 0; i < net->layers; i++) {
        for (int j = 0; j < net->neuronLayers[i]; j++) {
            free(net->W[i][j]);
        }
        free(net->W[i]);
        free(net->B[i]);
        free(net->Z[i]);
        free(net->A[i]);
        free(net->D[i]);
    }
    free(net->W);
    free(net->B);
    free(net->Z);
    free(net->A);
    free(net->D);

    printf("Memory Freed: Program Complete\n");
}