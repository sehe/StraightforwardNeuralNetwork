#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "DailyMinTemperatures.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class DailyMinTemperaturesTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        DailyMinTemperatures dataset("./datasets/daily-min-temperatures", 5);
        data = move(dataset.data);
    }
    
    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }
    
    static unique_ptr<Data> data;
};

unique_ptr<Data> DailyMinTemperaturesTest::data = nullptr;

TEST_F(DailyMinTemperaturesTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 1);
    ASSERT_EQ(data->numberOfLabel, 1);
    ASSERT_EQ(data->sets[training].inputs.size(), 3649);
    ASSERT_EQ(data->sets[training].labels.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 1);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 1);
    ASSERT_EQ(data->numberOfLabel, 1);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(2, 2),
        AllToAll(1, snn::identity)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(7_s || 1.6_mae);
    neuralNetwork.stopTraining();
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 1.6);
}