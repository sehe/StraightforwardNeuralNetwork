#include <thread>
#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Cifar10.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class Cifar10Test : public testing::Test
{
protected :
    Cifar10Test()
    {
        Cifar10 dataset;
        data = move(dataset.data);
    }

public :
    unique_ptr<Data> data;
};

TEST_F(Cifar10Test, DISABLED_loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 3072);
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 50000);
    ASSERT_EQ(data->sets[training].labels.size(), 50000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
}

TEST_F(Cifar10Test, DISABLED_trainNeuralNetwork)
{

}