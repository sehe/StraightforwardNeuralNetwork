#pragma once
#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"

namespace snn
{
    extern LayerModel AllToAll(int numberOfNeurons, activationFunction activation = sigmoid);

    extern LayerModel Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation = ReLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private :
        static std::unique_ptr<Layer> build(LayerModel model, int numberOfInputs, float* learningRate, float* momentum);

    public :
        static void build(std::vector<std::unique_ptr<Layer>>& layers, int numberOfInputs, std::vector<LayerModel>& models, float* learningRate, float* momentum);
    };
}