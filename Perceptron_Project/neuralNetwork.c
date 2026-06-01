#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// -------- USER ADJUSTED VARIABLES ---------
//Data
#define DATA_SIZE 20                //Amount of Data Points
#define INPUT_SIZE 3                //Number of Different Inputs / Parameters
#define TRAINING_SIZE 15            //How many data points to use for training
#define TESTING_SIZE (DATA_SIZE - TRAINING_SIZE)

//Training
#define EPOCHS 1000                  //Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.01            //How Fast Weights change based on Error
#define PRINT_INTERVAL 5           //How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50        //Minimum Epochs before Early Stopping can Occur
#define dropoutChance  0.5           //Chance to drop each neuron during training - 0 is 0%, 1 is 100% change of dropping
#define maxNorm 1.5                  //Maximum norm for weights if maxNormRegulation is enabled
#define momentumDecay 0.90           //Momentum factor
#define scalingDecay 0.9900          //Scaling factor for learning rate decay
#define clip 0.5f                    //Value to clip gradients for sigmoid activation function to prevent exploding gradients

//Features
bool earlyStopping = false;          //Whether to Stop Training if Error stops decreasing
bool dropout = false;                //Whether to randomly drop neurons during training to prevent overfitting
bool maxNormRegulation = true;      //Whether to cap weights to prevent exploding gradients and overfitting

//Optimizer
char optimizer = 'N';
//A = Adam Optimizer | Optimal Learning Rate = 0.0003
//R = RMSProp        | Optimal Learning Rate = 0.00005
//M = Momentum       | Optimal Learning Rate = 0.5
//N = None           | Optimal Learning Rate = 0.2

//Activation Function
char activationFunction = 'S';
//L = Leaky ReLU
//R = ReLU
//S = Sigmoid

typedef struct {
    float ***W;
    float **B;
    float ***Velocity;
    float **VelocityB;
    float ***Scaling;
    float **ScalingB;
    float **Z;
    float **A;
    float **D;
    float *maxValues;
} Network;

//Architecture
int neuronLayers[] = {50, 20, 1};    //Array of Neuron Counts for Each Layer

// Calculate layers
int layers = sizeof(neuronLayers) / sizeof(neuronLayers[0]);

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

//Function Prototypes
void testNetwork(Network *net);
void freeMemory(Network *net);
void normalizeData(Network *net);
void initializeParameters(Network *net);
float predictOutput(Network *net, float *Inputs);
Network* creatNetwork(int *neuronLayers);
void trainNetwork(Network *net);

int main() {

    //Create all variables for network
    Network *net = creatNetwork(neuronLayers);

    //Normalize inputs and labels
    normalizeData(net);

    //Randomly Initialize Weights and Biases using He Initialization
    initializeParameters(net);

    //Train Network
    trainNetwork(net);

    //Test Network
    testNetwork(net);

    //Try Predicting an Output
    float test[] = {0.5, 0.5, 0.5};
    float prediction = predictOutput(net, test);

    //Free memory when program is done
    freeMemory(net);

    return 0;
}

Network* creatNetwork(int *neuronLayers) {
    Network* net = malloc(sizeof(Network));

    //Max Values for Normalization
    net->maxValues = malloc(sizeof(float) * (INPUT_SIZE + 1));

    // ---------- Loop to declare all arrays needed to heap ----------
    //Weights
    net->W = malloc(sizeof(float **) * layers);

    //1st Moment / Velocity for each Weight
    net->Velocity = malloc(sizeof(float **) * layers);

    //1st Moment / Velocity for each Bias
    net->VelocityB = malloc(sizeof(float *) * layers);

    //2nd Moment / Scaling Factor for each Weight (for Adam Optimizer)
    net->Scaling = malloc(sizeof(float **) * layers);

    //2nd Moment / Scaling Factor for each Bias (for Adam Optimizer)
    net->ScalingB = malloc(sizeof(float *) * layers);

    //Bias
    net->B = malloc(sizeof(float *) * layers);

    //Preactiviation Value
    net->Z = malloc(sizeof(float *) * layers);

    //Activation Value
    net->A = malloc(sizeof(float *) * layers);

    //Delta for Backpropagation
    net->D = malloc(sizeof(float *) * layers);

    for (int i = 0; i < layers; i++) {

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
            net->W[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));
            net->Velocity[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));
            net->Scaling[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));

            //Set all Velocity and Scaling to 0
            net->ScalingB[i][j] = 0;
            net->VelocityB[i][j] = 0;
            for (int k = 0; k < ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]); k++) {
                net->Velocity[i][j][k] = 0;
                net->Scaling[i][j][k] = 0;
            }
        }
    }
    return net;
}

void normalizeData(Network *net) {

    // --- Find all maxes ---
    //Set max values to 0
    for (int i = 0; i < INPUT_SIZE + 1; i++) net->maxValues[i] = 0;

    //Loop through data to find max of each input and output
    for (int i = 0; i < DATA_SIZE; i++) {

        // Find max of inputs
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (x[i][j] > net->maxValues[j])
                net->maxValues[j] = x[i][j];
        }

        // Find max of outputs
        if (y[i] > net->maxValues[INPUT_SIZE])
            net->maxValues[INPUT_SIZE] = y[i];
    }

    // Loop through all data and divide by max
    for (int i = 0; i < DATA_SIZE; i++) {

        // Divide inputs by max
        for (int j = 0; j < INPUT_SIZE; j++)
            x[i][j] /= net->maxValues[j];

        // Divide output by max
        y[i] /= net->maxValues[INPUT_SIZE];
    }
    printf("Data Normalized\n");
}

void initializeParameters(Network *net) {

    // Seed Random Number Generator
    srand(time(NULL));

    int loops = 0;
    float scale = 0;

    // Initialize Weights and Bias with scaled random values (He Initialization)
    for (int i = 0; i < layers; i++) {

        // Inputs size loops for first layer, previous layer size for other layesr
        loops = (i == 0) ? INPUT_SIZE : neuronLayers[i - 1];

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
}

void trainNetwork(Network *net) {

    //All varibles needed later
    float eTotal = 0, eTrainingAvg = 0, lastEAvg = 1000, hiddenError = 0, scale = 0, currentGradient = 0, loops = 0, correctedA, correctedB = 0, weightLength = 0;

    //START TIMING
    clock_t start = clock();

    //Network Training Loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        //Reset Average Error for each Epoch
        eTrainingAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < TRAINING_SIZE; i++) {

            // --- Forward Pass ---
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Dropout - Randomly drop neurons during training to prevent overfitting
                    if (dropout && j < layers - 1) {

                        //Random chance to drop neuron
                        if (((float)rand() / RAND_MAX) < dropoutChance) {

                            //Set value to 0
                            net->A[j][k] = 0;

                            //Exit loop
                            continue;
                        }
                    }

                    //Set neuron value to bias
                    net->Z[j][k] = net->B[j][k];

                    //If first layer, add dot product of all inputs and weights
                    if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) net->Z[j][k] += x[i][z] * net->W[j][k][z];

                    //If not first layer, add dot product of all activations from previous layer and weights
                    else for (int z = 0; z < neuronLayers[j - 1]; z++) net->Z[j][k] += net->A[j - 1][z] * net->W[j][k][z];

                    //Activation Function
                    switch(activationFunction) {

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
            eTotal = y[i] - net->Z[layers - 1][0];

            //Calculate Average Error for Prints
            eTrainingAvg += fabs(eTotal / y[i]);

            // --- Backpropagation ---

            //Step 1: Calculate Blame for every neuron, starting with output layer

            //For output layer, delta is total error
            net->D[layers - 1][0] = eTotal;

            //For each layer
            for (int j = layers - 2; j >= 0; j--) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Set Delta to 0
                    net->D[j][k] = 0.0f;

                    //Sum of Deltas from layer ahead * corresponding weights
                    for (int z = 0; z < neuronLayers[j + 1]; z++) {
                        net->D[j][k] += net->D[j + 1][z] * net->W[j + 1][z][k];
                    }

                    //Activation Function Derivative
                    switch (activationFunction) {

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
                            if (net->D[j][k] > clip) net->D[j][k] = clip;
                            if (net->D[j][k] < -clip) net->D[j][k] = -clip;

                            break;
                        
                        //If no activation function, derivative = 1;
                    }
                }
            }

            //Step 2: Update Weights and Biases using Deltas
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    weightLength = 0;

                    //Update Bias
                    switch (optimizer) {

                        //Adam Optimizer
                        case 'A':

                            //Calculate 1st and 2nd moments
                            net->VelocityB[j][k] = momentumDecay * net->VelocityB[j][k] + (1 - momentumDecay) * net->D[j][k];
                            net->ScalingB[j][k] = scalingDecay * net->ScalingB[j][k] + (1 - scalingDecay) * net->D[j][k] * net->D[j][k];

                            //Bias Correction
                            correctedA = net->VelocityB[j][k] / (1 - pow(momentumDecay, epoch));
                            correctedB = net->ScalingB[j][k] / (1 - pow(scalingDecay, epoch));

                            //Update Bias
                            net->B[j][k] += LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                            break;

                        //RMSProp Optimizer
                        case 'R':
                            //Calculate 2nd moment
                            net->ScalingB[j][k] = scalingDecay * net->ScalingB[j][k] + (1 - scalingDecay) * net->D[j][k] * net->D[j][k];

                            //Update Bias
                            net->B[j][k] += LEARNING_RATE / (sqrt(net->ScalingB[j][k]) + 1e-8) * net->D[j][k];
                            break;

                        //Momentum
                        case 'M':

                            //Calculate Momentum
                            net->VelocityB[j][k] = momentumDecay * net->VelocityB[j][k] + (1 - momentumDecay) * net->D[j][k];
                            net->B[j][k] += net->VelocityB[j][k] * LEARNING_RATE;
                            break;
                        
                        //No Optimizer
                        default:
                            net->B[j][k] += LEARNING_RATE * net->D[j][k];
                            break;
                    }
                    
                    //Loops needed to update all weights
                    loops = (j == 0) ? INPUT_SIZE : neuronLayers[j - 1];

                    //For each weight
                    for (int z = 0; z < loops; z++) {
                        
                        //Calculate current gradient
                        currentGradient = (j == 0) ? net->D[j][k] * x[i][z] : net->D[j][k] * net->A[j - 1][z];

                        //Update Weights
                        switch (optimizer) {

                            //Adam Optimizer
                            case 'A':

                                //Calculate 1st and 2nd moments
                                net->Velocity[j][k][z] = momentumDecay * net->Velocity[j][k][z] + (1 - momentumDecay) * currentGradient;
                                net->Scaling[j][k][z] = scalingDecay * net->Scaling[j][k][z] + (1 - scalingDecay) * currentGradient * currentGradient;

                                //Bias Correction for Adam
                                correctedA = net->Velocity[j][k][z] / (1 - pow(momentumDecay, TRAINING_SIZE * epoch));
                                correctedB = net->Scaling[j][k][z] / (1 - pow(scalingDecay, TRAINING_SIZE * epoch));

                                //Update Weight with Adam
                                net->W[j][k][z] += LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                                break;

                            //RMSProp Optimizer
                            case 'R':
                                //Calculate 2nd moment
                                net->Scaling[j][k][z] = scalingDecay * net->Scaling[j][k][z] + (1 - scalingDecay) * currentGradient * currentGradient;

                                //Update Weights
                                net->W[j][k][z] += LEARNING_RATE / (sqrt(net->Scaling[j][k][z]) + 1e-8) * currentGradient;
                                break;

                            //Momentum
                            case 'M':

                                //Calculate Momentum
                                net->Velocity[j][k][z] = momentumDecay * net->Velocity[j][k][z] + (1 - momentumDecay) * currentGradient;

                                //Update Weights
                                net->W[j][k][z] += net->Velocity[j][k][z] * LEARNING_RATE;
                                break;

                            //No Optimizer
                            default:
                                net->W[j][k][z] += LEARNING_RATE * currentGradient;
                                break;
                        }

                        //Max Norm Regulation - Limit the maximum norm of the weights to prevent exploding gradients
                        if (maxNormRegulation) weightLength += net->W[j][k][z] * net->W[j][k][z];
                    }

                    if (maxNormRegulation) {
                        weightLength = sqrt(weightLength);
                        if (weightLength > maxNorm) {
                            for (int z = 0; z < loops; z++) {
                                net->W[j][k][z] = maxNorm;
                            }
                        }
                    }
                }
            }
        }

        //Calculate Average Error for Epoch
        eTrainingAvg = eTrainingAvg / TRAINING_SIZE * 100;

        //Mark continuous timing and calculate total runtime
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % PRINT_INTERVAL == 0) {

            //Print Epoch, Average Error, Runtime, and Result
            printf("Epoch: %i | Average Error: %.4f | Runtime: %.1f | Result: %.3f\n", epoch, eTrainingAvg, runtime * 1000, net->Z[layers - 1][0]);

            //for (int j = 0; j < layers; j++) {
            //    for (int k = 0; k < neuronLayers[j]; k++) {
            //        printf("D[%d][%d] = %.6f | W[%d][%d][0] = %.6f\n", j, k, D[j][k], j, k, W[j][k][0]);
            //    }
            //}
        }

        //Stop early if error improvement is less than 0.001 and using early stopping
        if (earlyStopping && lastEAvg - eTrainingAvg < 0.001 && epoch > MIN_STOPPING_EPOCH) {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eTrainingAvg);
            break;
        }

        //Update lastEAvg for Early Stopping Check
        lastEAvg = eTrainingAvg;
    }
}

void testNetwork(Network*net) {

    float eTestingAvg = 0;

    //Loop through testing data
    for (int i = TRAINING_SIZE; i < DATA_SIZE; i++) {

        // For each layer
        for (int j = 0; j < layers; j++) {

            // For each neuron
            for (int k = 0; k < neuronLayers[j]; k++) {

                //Set neuron value to bias
                net->Z[j][k] = net->B[j][k];

                //If first layer, use inputs
                if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) net->Z[j][k] += x[i][z] * net->W[j][k][z];

                //Else, use outputs from previous layer
                else for (int z = 0; z < neuronLayers[j - 1]; z++) net->Z[j][k] += net->A[j - 1][z] * net->W[j][k][z];

                //Activation Function
                switch(activationFunction) {

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
        eTestingAvg += fabs((y[i] - net->Z[layers - 1][0]) / y[i]);
    }

    //Calculate Average Error for Epoch
    eTestingAvg = (eTestingAvg / TESTING_SIZE) * 100;

    printf("Testing Average Error: %.2f\n", eTestingAvg);
}

float predictOutput(Network *net, float *Inputs) {
 
    //For each layer
    for (int j = 0; j < layers; j++) {

        //For each neuron
        for (int k = 0; k < neuronLayers[j]; k++) {

            //Set neuron value to bias
            net->Z[j][k] = net->B[j][k];

            //If first layer, use inputs
            if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) net->Z[j][k] += Inputs[z] * net->W[j][k][z];

            //Else, use outputs from previous layer
            else for (int z = 0; z < neuronLayers[j - 1]; z++) net->Z[j][k] += net->A[j - 1][z] * net->W[j][k][z];

            //Activation Function
            switch(activationFunction) {

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
    printf("Predicted Value: %.3f\n", net->Z[layers - 1][0]);

    //Return Final Output
    return net->Z[layers - 1][0];
}

void freeMemory(Network *net) {

    //Free Allocated Memory
    for (int i = 0; i < layers; i++) {
        for (int j = 0; j < neuronLayers[i]; j++) {
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