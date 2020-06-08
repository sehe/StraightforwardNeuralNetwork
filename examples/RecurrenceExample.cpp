#include "Examples.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a time series.
In this neural network return the sum of 2 inputs.
For more explanation go to wiki.
*/
int recurrenceExample()
{
    vector<vector<float>> inputData = {
        {3}, {5}, {4}, {2}, {0}, {2}, {4}, {1}, {4}, {3}, {0}, {0}, {4}, {3}, {2}, {1}, {2}, {0}, {1}, {5}, {5}, {3}
    };
    vector<vector<float>> expectedOutputs = {
        {0.3}, {0.8}, {0.9}, {0.6}, {0.2}, {0.2}, {0.6}, {0.5}, {0.5}, {0.7}, {0.3}, {0.0}, {0.4}, {0.7}, {0.5}, {0.3}, {0.3}, {0.2}, {0.1}, {0.6}, {1.0}, {0.8}
    };

    const float precision = 0.05f;
    Data data(regression, inputData, expectedOutputs, continuous, 1);
    data.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(6, 1),
        AllToAll(1, snn::tanh)
    });
    neuralNetwork.optimizer.learningRate = 0.0003f;
    //neuralNetwork.optimizer.momentum = 0.98f;

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc /*|| 8_s*/); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
    float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100
        && mae < precision
        && neuralNetwork.isValid() == 0)
    {
        return EXIT_SUCCESS; // the neural network has learned
    }
    return EXIT_FAILURE;
}