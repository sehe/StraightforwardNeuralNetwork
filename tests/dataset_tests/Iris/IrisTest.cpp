#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Iris.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class IrisTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        Iris dataset;
        data = move(dataset.data);
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> IrisTest::data = nullptr;

TEST_F(IrisTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 4);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 150);
    ASSERT_EQ(data->sets[training].labels.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 150);
}

TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        4,
        {
            AllToAll(15),
            AllToAll(5),
            AllToAll(3)
        });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(0.98_acc || 2_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98);
}