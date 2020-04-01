#include <boost/serialization/export.hpp>
#include "Convolution2D.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution2D)

Convolution2D::Convolution2D(LayerModel& model, StochasticGradientDescent* optimizer)
    : Convolution(model, optimizer)
{
}

inline
unique_ptr<Layer> Convolution2D::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<Convolution2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

vector<float> Convolution2D::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

std::vector<float> Convolution2D::backOutput(std::vector<float>& inputErrors)
{
    //TODO: adapt for convolution
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& result = neurons[n].backOutput(inputErrors[n]);
        for (int r = 0; r < numberOfInputs; ++r)
            errors[r] += result[r];
    }
    return {}; //errors;
}

void Convolution2D::train(std::vector<float>& inputErrors)
{
    throw NotImplementedException();
}

std::vector<int> Convolution2D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfConvolutionMatrix - 1),
        this->shapeOfInput[1] - (this->sizeOfConvolutionMatrix - 1),
        this->numberOfConvolution
    };
}

int Convolution2D::isValid() const
{
    return this->Convolution::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    vector<float> neuronInputs{};

    const int neuronPositionX = neuronNumber % this->shapeOfInput[0];
    const int neuronPositionY = neuronNumber / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfConvolutionMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[2] * this->shapeOfInput[0]) + neuronPositionX * this->shapeOfInput[2];
        const int endIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2])
        + (neuronPositionX + this->sizeOfConvolutionMatrix) * this->shapeOfInput[2];
        for (int j = beginIndex; j < endIndex; ++j)
        {
            neuronInputs.push_back(inputs[j]);
        }
    }
    return neuronInputs;
}

inline
bool Convolution2D::operator==(const Convolution2D& layer) const
{
    return this->Convolution::operator==(layer);
}

inline
bool Convolution2D::operator!=(const Convolution2D& layer) const
{
    return !(*this == layer);
}
