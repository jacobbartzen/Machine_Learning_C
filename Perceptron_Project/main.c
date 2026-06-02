#include "neuralNetwork.h"

//Data
#define DATA_SIZE 20                //Amount of Data Points
#define INPUT_SIZE 3                //Number of Different Inputs / Parameters
#define TRAINING_SIZE 15            //How many data points to use for training

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