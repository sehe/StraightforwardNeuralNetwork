#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Recurrence, RepeatInput)
{
    vector2D<float> inputData =       {{0.0}, {-1.0}, {-0.8}, {-0.5}, {-0.2}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}};
    vector2D<float> expectedOutputs = {{0.0}, {-1.0}, {-0.8}, {-0.5}, {-0.2}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}};
    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.15);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12),
        FullyConnected(6),
        FullyConnected(1, activation::tanh)
    });
    neuralNetwork.optimizer.learningRate = 0.03f;
    neuralNetwork.optimizer.momentum = 0.97f;
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

TEST(Recurrence, RepeatLastInput)
{
    vector2D<float> inputData =       {{0}, {0}, {1}, {1}, {0}, {-1}, {-1}, {0},  {1}, {-1}, {1},  {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {1}, {1}, {0},  {-1}, {-1}, {0}, {1},  {-1}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.4);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(20),
        FullyConnected(8),
        FullyConnected(1, activation::tanh)
    });
    neuralNetwork.optimizer.learningRate = 0.02f;
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

//a simple recurrent neural network can't solve this problem
TEST(Recurrence, RepeatLastLastInput)
{
    vector2D<float> inputData =       {{0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {1}, {0}, {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 2);
    data->setPrecision(0.3);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        GruLayer(15),
        GruLayer(5),
        FullyConnected(1)
    });
    neuralNetwork.optimizer.learningRate = 0.05f;
    neuralNetwork.optimizer.momentum = 0.1f;
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

inline
void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(1.0_acc || 7_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.5);
}
