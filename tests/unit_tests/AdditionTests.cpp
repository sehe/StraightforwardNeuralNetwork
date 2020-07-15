#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForAdditionTests();
unique_ptr<Data> createRecurrentDataForAdditionTests();
void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Addition, WithMPL)
{
    unique_ptr<Data> data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        FullyConnected(6, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithCNN)
{
    auto data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        Convolution(6, 1, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithRNN)
{
    auto data = createRecurrentDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(6, 1, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(1.0_acc || 15_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.5);
}

unique_ptr<Data> createDataForAdditionTests()
{
    vector2D<float> inputData = {
        {3, 5}, {5, 4}, {4, 2}, {2, 0}, {0, 2},
        {2, 4}, {4, 1}, {1, 4}, {4, 3}, {3, 0},
        {0, 0}, {0, 4}, {4, 3}, {3, 2}, {2, 1},
        {1, 2}, {2, 0}, {0, 1}, {1, 2}, {5, 5},
        {5, 3}, {1, 1}, {4, 4}, {3, 3}, {2, 2}
    };
    vector2D<float> expectedOutputs = {
        {8}, {9}, {6}, {2}, {2},
        {6}, {5}, {5}, {7}, {3},
        {0}, {4}, {7}, {5}, {3},
        {3}, {2}, {1}, {3}, {10},
        {8}, {2}, {8}, {6}, {4}
    };

    const float precision = 0.5f;
    unique_ptr<Data> data = make_unique<Data>(regression, inputData, expectedOutputs);
    data->setPrecision(precision);
    return data;
}

unique_ptr<Data> createRecurrentDataForAdditionTests()
{
    vector2D<float> inputData = {
        {3}, {5}, {4}, {2}, {0}, {2}, {2}, {4}, {1}, {4}, {3}, {0}, {0}, {4}, {4}, {3}, {2}, {1}, {2}, {0}, {1}, {5}, {5}, {3}, {3}
    };
    vector2D<float> expectedOutputs = {
        {3}, {8}, {9}, {6}, {2}, {2}, {4}, {6}, {5}, {5}, {7}, {3}, {0}, {4}, {8}, {7}, {5}, {3}, {3}, {2}, {1}, {6}, {10}, {8}, {6}
    };

    const float precision = 0.5f;
    auto data = make_unique<Data>(regression, inputData, expectedOutputs, timeSeries, 1);
    data->setPrecision(precision);
    return data;
}