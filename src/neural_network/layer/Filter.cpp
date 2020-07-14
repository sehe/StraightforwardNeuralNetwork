#include <boost/serialization/export.hpp>
#include "Filter.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Filter)

Filter::Filter(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

vector<float> Filter::output(const vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(neuronInputs);
    }
    return outputs;
}

vector<float> Filter::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = neurons[n].backOutput(inputErrors[n]);
        this->insertBackOutputForNeuron(n, error, errors);
    }
    return errors;
}

int Filter::isValid() const
{
    return this->Layer::isValid();
}

bool Filter::operator==(const BaseLayer& layer) const
{
   try
    {
        const auto& f = dynamic_cast<const Filter&>(layer);
        return this->Layer::operator==(layer)
            && this->numberOfInputs == f.numberOfInputs
            && this->errors == f.errors
            && this->neurons == f.neurons;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool Filter::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}