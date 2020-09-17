#include "Examples.hpp"
#include "../src/neural_network/StraightforwardNeuralNetwork.hpp"
#include "../src/data/Data.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a classification problem.
The neural network return class 0 if sum of inputs is negative and class 1 il sum of input is positive.
For more explanation go to wiki.
*/
int classificationExample()
{
    vector<vector<float>> inputData = {{-0.1f, 0.4f, -0.6f}, {0.5f, -0.4f, -0.8f}, {-0.7f, 0.9f, -0.7f}, {-0.9f, -0.5f, 1.7f}, {0.5f, -0.5f, 0.9f}, {0.3f, 0.6f, 0.8f}};
    vector<vector<float>> expectedOutputs = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}};

    Data data(problem::classification, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork({Input(3), FullyConnected(5), FullyConnected(2)});

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc || 3_s ); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
    int classNumber = neuralNetwork.computeCluster(data.getData(snn::testing, 0)); // consult neural network to test it
    int expectedClassNumber = data.getLabel(snn::testing, 0); // return position of neuron with highest output
    if (accuracy == 100
    && classNumber == expectedClassNumber
    && neuralNetwork.isValid() == 0)
    {
        return EXIT_SUCCESS; // the neural network has learned
    }
    return EXIT_FAILURE;
}