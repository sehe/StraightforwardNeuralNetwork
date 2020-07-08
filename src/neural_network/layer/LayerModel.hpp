#pragma once
#include <vector>
#include "LayerType.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn
{
    struct LayerModel
    {
        layerType type;
        activationFunction activation;
        int numberOfInputs;
        int numberOfNeurons;
        int numberOfInputsByNeurons;
        int numberOfRecurrences;
        int numberOfFilters;
        int sizeOfFilerMatrix;
        std::vector<int> shapeOfInput;
    };
}