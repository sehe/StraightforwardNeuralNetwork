#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Cifar10.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class Cifar10Test : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        Cifar10  dataset("./datasets/CIFAR-10");
        data = move(dataset.data);
    }

    void SetUp() override 
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> Cifar10Test::data = nullptr;

TEST_F(Cifar10Test, loadData)
{
    ASSERT_EQ(data->sizeOfData, 3072);
    ASSERT_EQ(data->numberOfLabels, 10);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 50000);
    ASSERT_EQ((int)data->sets[training].labels.size(), 50000);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(Cifar10Test, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3072),
        FullyConnected(100),
        FullyConnected(25),
        FullyConnected(10)
    });
    neuralNetwork.train(*data, 1_ep || 240_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.24f);
}

TEST_F(Cifar10Test, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(32, 32, 3),
        Convolution(1,8),
        FullyConnected(100),
        FullyConnected(10)
    },
        StochasticGradientDescent(0.03f, 0.2f));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForCIFAR-10.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*data, 10_ep);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20f);
}

TEST_F(Cifar10Test, EvaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./BestNeuralNetworkForCIFAR-10.snn");
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_FLOAT_EQ(accuracy, 0.5225f);
}