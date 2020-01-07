#include <thread>
#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Wine.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class WineTest : public testing::Test
{
protected :
    WineTest()
    {
        Wine dataset;
        data = move(dataset.data);
    }

public :
    shared_ptr<Data> data;
};

TEST_F(WineTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 13);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 178);
    ASSERT_EQ(data->sets[training].labels.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 178);
}

TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        13,
        {
            AllToAll(20),
            AllToAll(8),
            AllToAll(3)
        });
    neuralNetwork.trainingStart(*data);
    this_thread::sleep_for(3s);
    neuralNetwork.trainingStop();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}