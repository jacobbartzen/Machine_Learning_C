#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// -------- USER ADJUSTED VARIABLES ---------
//Data
#define DATA_SIZE 20          // Amount of Data Points
#define INPUT_SIZE 3          // Number of Different Inputs
#define TRAINING_SIZE 15      // How much of data to use for training
int TESTING_SIZE = DATA_SIZE - TRAINING_SIZE;

//Training
#define EPOCHS 5000        //Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.0003     //How Fast Weights change based on Error
#define PRINT_INTERVAL 1000  //How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50 //Minimum Epochs before Early Stopping can Occur
float dropoutChance = 0.05;   //Chance to drop each neuron during training - 0 is 0%, 1 is 100% change of dropping
float maxNorm = 0.6;          //Maximum norm for weights if maxNormRegulation is enabled
#define momentumDecay 0.90    // Momentum factor
#define scalingDecay 0.999     // Scaling factor for learning rate decay

//Features
bool earlyStopping = false;        //Whether to Stop Training if Error stops decreasing
bool dropout = false;              //Whether to randomly drop neurons during training to prevent overfitting
bool maxNormRegulation = false;    //Whether to cap weights to prevent exploding gradients and overfitting

//Optimizers (ONLY SET 1 TO TRUE)     //Optimal Learning Rate for Optimizer
bool adamOptimizer = true;            //Learning Rate: 0.0003
bool RMSPropOptimizer = false;        //Learning Rate: 0.00005
bool momentumOptimizer = false;       //Learning Rate: 0.5
//No Opimizer                         //Learning Rate: 0.2

//Architecture
int neuronLayers[] = {50, 20, 1};  //Array of Neuron Counts for Each Layer

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

int main() {

    //Max Values for Normalization
    float maxValues[INPUT_SIZE + 1];

    int layers = sizeof(neuronLayers) / sizeof(neuronLayers[0]);

    // ---------- Loop to declare all arrays needed to heap ----------
    //Weights
    float ***W = malloc(sizeof(float**) * layers);

    //1st Moment / Velocity for each Weight
    float ***Velocity = malloc(sizeof(float**) * layers);

    //1st Moment / Velocity for each Bias
    float **VelocityB = malloc(sizeof(float*) * layers);

    //2nd Moment / Scaling Factor for each Weight (for Adam Optimizer)
    float ***Scaling = malloc(sizeof(float**) * layers);

    //2nd Moment / Scaling Factor for each Bias (for Adam Optimizer)
    float **ScalingB = malloc(sizeof(float*) * layers);

    //Bias
    float **B = malloc(sizeof(float*) * layers);

    //Preactiviation Value
    float **Z = malloc(sizeof(float*) * layers);

    //Activation Value
    float **A = malloc(sizeof(float*) * layers);

    //Delta for Backpropagation
    float **D = malloc(sizeof(float*) * layers);

    for (int i = 0; i < layers; i++) {

        //Allocate Memory for Each Layer
        W[i] = malloc(sizeof(float*) * neuronLayers[i]);
        Velocity[i] = malloc(sizeof(float*) * neuronLayers[i]);
        VelocityB[i] = malloc(sizeof(float) * neuronLayers[i]);
        Scaling[i] = malloc(sizeof(float*) * neuronLayers[i]);
        ScalingB[i] = malloc(sizeof(float) * neuronLayers[i]);
        B[i] = malloc(sizeof(float) * neuronLayers[i]);
        Z[i] = malloc(sizeof(float) * neuronLayers[i]);
        A[i] = malloc(sizeof(float) * neuronLayers[i]);
        D[i] = malloc(sizeof(float) * neuronLayers[i]);

        //Create 3d array for Weights and Velocity
        for (int j = 0; j < neuronLayers[i]; j++) {
            W[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));
            Velocity[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));
            Scaling[i][j] = malloc(sizeof(float) * ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]));

            //Set all Velocity and Scaling to 0
            ScalingB[i][j] = 0;
            VelocityB[i][j] = 0;
            for (int k = 0; k < ((i == 0) ? INPUT_SIZE : neuronLayers[i - 1]); k++) {
                Velocity[i][j][k] = 0;
                Scaling[i][j][k] = 0;
            }
        }
    }

    //All varibles needed later
    float eTotal = 0, eTrainingAvg = 0, lastEAvg = 1000, eTestingAvg = 0, hiddenError = 0, scale = 0, currentGradient = 0, loops = 0, correctedA, correctedB = 0;

    //Find all maxes
    for (int i = 0; i < INPUT_SIZE + 1; i++) maxValues[i] = 0;
    for (int i = 0; i < DATA_SIZE; i++) {

        //Find max of inputs
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (x[i][j] > maxValues[j]) maxValues[j] = x[i][j];
        }

        //Find max of outputs
        if (y[i] > maxValues[INPUT_SIZE]) maxValues[INPUT_SIZE] = y[i];
    }

    //Divide data by max
    for (int i = 0; i < DATA_SIZE; i++) {

        //Divide inputs by max
        for (int j = 0; j < INPUT_SIZE; j++) {
            x[i][j] /= maxValues[j];
        }

        //Divide output by max
        y[i] /= maxValues[INPUT_SIZE];
    }

    //Seed Random Number Generator
    srand(time(NULL));

    //Initialize Weights and Bias with small random values between -0.005 and 0.005
    for (int i = 0; i < layers; i++) {
        int prevSize = (i == 0) ? INPUT_SIZE : neuronLayers[i - 1];
        scale = sqrt(2.0f / prevSize);
        for (int j = 0; j < neuronLayers[i]; j++) {
            B[i][j] = 0; // bias initialized to 0 is standard with He init
            for (int k = 0; k < prevSize; k++) {
                W[i][j][k] = (((float)rand() / RAND_MAX) * 2 - 1) * scale;
            }
        }
    }

    printf("Weights and Biases Allocated and Randomly Initialized - Starting Training\n");

    //START TIMING
    clock_t start = clock();

    //Network Training Loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        //Reset Average Error for Epoch
        eTrainingAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < TRAINING_SIZE; i++) {

            //Forward Pass
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Dropout - Randomly drop neurons during training to prevent overfitting
                    if (dropout && j < layers - 1) {
                        if (((float)rand() / RAND_MAX) < dropoutChance) { // 20% chance to drop
                            A[j][k] = 0; // Drop neuron by setting activation to 0
                            continue; // Skip calculations for this neuron
                        }
                    }

                    Z[j][k] = B[j][k];

                    //If first layer, use inputs
                    if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) Z[j][k] += x[i][z] * W[j][k][z];

                    //Else, use outputs from previous layer
                    else for (int z = 0; z < neuronLayers[j - 1]; z++) Z[j][k] += A[j - 1][z] * W[j][k][z];

                    //Activation Function: Leaky ReLU (Rectified Linear Unit)
                    A[j][k] = (Z[j][k] > 0) ? Z[j][k] : 0.01f * Z[j][k];
                }
            }

            //Calculate Total Error
            eTotal = y[i] - Z[layers - 1][0];

            //Calculate Abs Average Error
            eTrainingAvg += fabs(eTotal / y[i]);

            //Backward Pass (Backpropagation)
            //For output layer, delta is total error
            D[layers - 1][0] = eTotal;

            //For each layer
            for (int j = layers - 2; j >= 0; j--) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Set Delta to 0
                    D[j][k] = 0.0f;

                    //Sum of Deltas from layer ahead * corresponding weights
                    for (int z = 0; z < neuronLayers[j + 1]; z++) {
                        D[j][k] += D[j + 1][z] * W[j + 1][z][k];
                    }

                    //Leaky ReLU Derivative
                    D[j][k] *= (Z[j][k] > 0) ? 1.0f : 0.01f;
                }
            }

            //Update Weights and Biases using Deltas
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    if (momentumOptimizer || RMSPropOptimizer || adamOptimizer) {

                        //Calculate Bias Velocity / 1st Moment
                        VelocityB[j][k] = momentumDecay * VelocityB[j][k] + (1 - momentumDecay) * D[j][k];

                        if (momentumOptimizer) {
                            //Update Bias
                            B[j][k] += VelocityB[j][k] * LEARNING_RATE;
                        }

                        else if (RMSPropOptimizer || adamOptimizer) {

                            //Calculate Bias Scaling / 2nd Moment
                            ScalingB[j][k] = scalingDecay * ScalingB[j][k] + (1 - scalingDecay) * D[j][k] * D[j][k];

                            if (RMSPropOptimizer) {
                                //Update Bias with RMSProp
                                B[j][k] += LEARNING_RATE / (sqrt(ScalingB[j][k]) + 1e-8) * D[j][k];
                            }

                            else if (adamOptimizer) {
                                //Bias Correction for Adam
                                correctedA = VelocityB[j][k] / (1 - pow(momentumDecay, epoch));
                                correctedB = ScalingB[j][k] / (1 - pow(scalingDecay, epoch));

                                //Update Bias with Adam
                                B[j][k] += LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                            }
                        }
                    }
                    else B[j][k] += LEARNING_RATE * D[j][k];

                    loops = (j == 0) ? INPUT_SIZE : neuronLayers[j - 1];

                    for (int z = 0; z < loops; z++) {
                        
                        currentGradient = (j == 0) ? D[j][k] * x[i][z] : D[j][k] * A[j - 1][z];
                        
                        if (momentumOptimizer || RMSPropOptimizer || adamOptimizer) {
                            //Calculate Momentum Velocity / 1st Moment
                            Velocity[j][k][z] = momentumDecay * Velocity[j][k][z] + (1 - momentumDecay) * currentGradient;
                            
                            if (momentumOptimizer) {
                                //Update Weight with Momentum
                                W[j][k][z] += Velocity[j][k][z] * LEARNING_RATE;
                            }

                            else if (RMSPropOptimizer || adamOptimizer) {
                                //Calculate Scaling / 2nd Moment
                                Scaling[j][k][z] = scalingDecay * Scaling[j][k][z] + (1 - scalingDecay) * currentGradient * currentGradient;

                                if (RMSPropOptimizer) {
                                    // Update Weight with RMSProp
                                    W[j][k][z] += LEARNING_RATE / (sqrt(Scaling[j][k][z]) + 1e-8) * currentGradient;
                                }

                                else if (adamOptimizer) {
                                    //Bias Correction for Adam
                                    correctedA = Velocity[j][k][z] / (1 - pow(momentumDecay, epoch));
                                    correctedB = Scaling[j][k][z] / (1 - pow(scalingDecay, epoch));

                                    //Update Weight with Adam
                                    W[j][k][z] += LEARNING_RATE / (sqrt(correctedB) + 1e-8) * correctedA;
                                }
                            }
                        }
                        else W[j][k][z] += LEARNING_RATE * currentGradient;

                        //Max Norm Regulation - Limit the maximum norm of the weights to prevent exploding gradients
                        if (maxNormRegulation && fabs(W[j][k][z]) > maxNorm) W[j][k][z] = maxNorm;
                    }
                }
            }
        }

        //Calculate Average Error for Epoch
        eTrainingAvg = (eTrainingAvg / TRAINING_SIZE) * 100;

        //End timing and calculate total runtime for epoch
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % PRINT_INTERVAL == 0) {

            //Print Epoch, Average Error, and Runtime
            printf("Epoch: %i | Average Error: %.4f | Runtime: %.1f | Result: %.3f\n", epoch, eTrainingAvg, runtime * 1000, Z[layers - 1][0]);

            //for (int j = 0; j < layers; j++) {
            //    for (int k = 0; k < neuronLayers[j]; k++) {
            //        printf("D[%d][%d] = %.6f | W[%d][%d][0] = %.6f\n", j, k, D[j][k], j, k, W[j][k][0]);
            //    }
            //}
        }

        //Stop early if error improvement is less than 0.001
        if (earlyStopping && lastEAvg - eTrainingAvg < 0.001 && epoch > MIN_STOPPING_EPOCH) {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eTrainingAvg);
            break;
        }

        lastEAvg = eTrainingAvg;
    }

    //Loop through testing data
    for (int i = TRAINING_SIZE; i < DATA_SIZE; i++) {

        // For each layer
        for (int j = 0; j < layers; j++) {

            // For each neuron
            for (int k = 0; k < neuronLayers[j]; k++) {

                Z[j][k] = B[j][k];

                // If first layer, use inputs
                if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) Z[j][k] += x[i][z] * W[j][k][z];

                // Else, use outputs from previous layer
                else for (int z = 0; z < neuronLayers[j - 1]; z++) Z[j][k] += A[j - 1][z] * W[j][k][z];

                // Activation Function: ReLU (Rectified Linear Unit)
                A[j][k] = (Z[j][k] > 0) ? Z[j][k] : 0;
            }
        }

        // Calculate Abs Average Error
        eTestingAvg += fabs((y[i] - Z[layers - 1][0]) / y[i]);
    }

    // Calculate Average Error for Epoch
    eTestingAvg = (eTestingAvg / TESTING_SIZE) * 100;

    printf("Training Final Error: %.2f | Testing Average Error: %.2f\n", eTrainingAvg, eTestingAvg);

    //Free Allocated Memory
    for (int i = 0; i < layers; i++) {
        for (int j = 0; j < neuronLayers[i]; j++) {
            free(W[i][j]);
        }
        free(W[i]);
        free(B[i]);
        free(Z[i]);
        free(A[i]);
        free(D[i]);
    }
    free(W);
    free(B);
    free(Z);
    free(A);
    free(D);

    printf("Memory Freed: Program Complete\n");

    return 0;
}