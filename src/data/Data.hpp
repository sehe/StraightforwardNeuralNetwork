#pragma once
#include <vector>
#include "../tools/Tools.hpp"

namespace snn
{
    enum set
    {
        testing = 0,
        training = 1
    };

    enum temporalType
    {
        nonTemporal,
        temporal,
        continuous,
    };

    class Data
    {
    private:
        void initialize(std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels,
                        std::vector<std::vector<float>>& testingInputs,
                        std::vector<std::vector<float>>& testingLabels,
                        float value,
                        temporalType type);

        void shuffleNonTemporal();
        void shuffleTemporal();
        void shuffleContinuous();

        temporalType type;

    protected:
        std::vector<int> indexes;
        float value;
        void clearData();

        Data(std::vector<std::vector<float>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<float>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             float value,
             temporalType type);

        Data(std::vector<std::vector<float>>& inputs,
             std::vector<std::vector<float>>& labels,
             float value,
             temporalType type);

        Data(std::vector<std::vector<std::vector<float>>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<std::vector<float>>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             float value,
             temporalType temporalType);

        Data(std::vector<std::vector<std::vector<float>>>& inputs,
             std::vector<std::vector<float>>& labels,
             float value,
             temporalType temporalType);

    public:
        int sizeOfData; // size of one data, equal to size of neural network inputs
        int numberOfLabel; // the number of class, equal to size of neural network outputs

        struct Set
        {
            int index{0};
            int size{0}; // number of data inside set
            vector3D<float> inputs{};
            vector2D<float> labels{};
        } sets[2];

        virtual ~Data() = default;

        void normalization(float min, float max);

        void shuffle();
        void unshuffle();

        [[nodiscard]] bool isValid();

        [[nodiscard]] float getValue() const { return value; }

        [[nodiscard]] virtual const std::vector<float>& getTrainingData(const int index);
        [[nodiscard]] virtual const std::vector<float>& getTestingData(const int index);

        [[nodiscard]] virtual int getTrainingLabel(const int) { throw std::exception(); }
        [[nodiscard]] virtual int getTestingLabel(const int) { throw std::exception(); }

        [[nodiscard]] virtual const std::vector<float>& getTrainingOutputs(const int index);
        [[nodiscard]] virtual const std::vector<float>& getTestingOutputs(const int) = 0;

        [[nodiscard]] const std::vector<float>& getData(set set, const int index);
        [[nodiscard]] const std::vector<float>& getOutputs(set set, const int index);
        [[nodiscard]] int getLabel(set set, const int index);
    };
}
