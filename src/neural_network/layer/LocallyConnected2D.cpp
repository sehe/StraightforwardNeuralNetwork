#include <boost/serialization/export.hpp>
#include "LocallyConnected2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected2D)

LocallyConnected2D::LocallyConnected2D(LayerModel& model, StochasticGradientDescent* optimizer)
    : Filter(model, optimizer)
{
}

inline
unique_ptr<Layer> LocallyConnected2D::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<LocallyConnected2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

std::vector<int> LocallyConnected2D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfFilterMatrix - 1),
        this->shapeOfInput[1] - (this->sizeOfFilterMatrix - 1),
        this->numberOfFilters
    };
}

int LocallyConnected2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->sizeOfFilterMatrix * this->shapeOfInput[2])
            return 203;
    }
    return this->Filter::isValid();
}

inline
vector<float> LocallyConnected2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    vector<float> neuronInputs;
    neuronInputs.reserve(this->neurons[neuronNumber].getNumberOfInputs());
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfFilters;
    const int neuronPositionX = neuronNumber % (this->shapeOfInput[0] * this->sizeOfFilterMatrix);
    const int neuronPositionY = neuronNumber / (this->shapeOfInput[0] * this->sizeOfFilterMatrix);

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPositionX * this->shapeOfInput[2];
        const int endIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2])
        + (neuronPositionX + this->sizeOfFilterMatrix) * this->shapeOfInput[2];
        for (int j = beginIndex; j < endIndex; ++j)
        {
            neuronInputs.push_back(inputs[j]);
        }
    }
    return neuronInputs;
}

void LocallyConnected2D::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const
{
    const int neuronPositionX = neuronNumber % (this->shapeOfInput[0] * this->sizeOfFilterMatrix);
    const int neuronPositionY = neuronNumber / (this->shapeOfInput[0] * this->sizeOfFilterMatrix);

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[2] * this->shapeOfInput[0]) + neuronPositionX * this->shapeOfInput[2];
        for(int e = 0; e < error.size(); ++e)
        {
            const int index = beginIndex + e;
            errors[index] += error[e];
        }
    }
}

inline
bool LocallyConnected2D::operator==(const LocallyConnected2D& layer) const
{
    return this->Filter::operator==(layer);
}

inline
bool LocallyConnected2D::operator!=(const LocallyConnected2D& layer) const
{
    return !(*this == layer);
}
