#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForOptimisezTests(int numberOfData, int sizeOfData);

TEST(Optimizer, FindRightValueIn20)
{
    unique_ptr<Data> data = createDataForOptimisezTests(1000, 20);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(20),
        FullyConnected(4, activation::tanh),
        FullyConnected(1, activation::sigmoid)
    },
        StochasticGradientDescent(0.01f, 0.4f));

    neuralNetwork.train(*data, 1.00_acc || 3_s);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    auto acc = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0f);
    ASSERT_MAE(mae, 0.6f);
}

unique_ptr<Data> createDataForOptimisezTests(int numberOfData, int sizeOfData)
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;

    inputData.reserve(numberOfData);
    expectedOutputs.reserve(numberOfData);
    for(int i = 0; i < numberOfData; ++i)
    {
        inputData.push_back(vector<float>());
        inputData.back().reserve(sizeOfData);
        for(int j = 0; j < sizeOfData; ++j)
        {
            const float rand = internal::Tools::randomBetween(-1.0f, 1.0f);
            inputData.back().push_back(rand);
        }
        if(inputData[i][0] > 0)
            expectedOutputs.push_back({1.0f});
        else
            expectedOutputs.push_back({0.0f});
    }
    unique_ptr<Data> data = make_unique<Data>(problem::regression, inputData, expectedOutputs);
    data->setPrecision(0.5);
    return data;
}
