#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(Neuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer),
      numberOfRecurrences(model.numberOfRecurrences),
      numberOfInputs(model.numberOfInputs - this->numberOfRecurrences),
      sizeOfInputs(sizeof(float) * model.numberOfInputs - this->numberOfRecurrences),
      sizeToCopy(sizeof(float) * model.numberOfRecurrences - 1),
      indexEnd(model.numberOfRecurrences - 2)
{
    this->recurrences.resize(model.numberOfRecurrences, -1);
}

float RecurrentNeuron::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset)
        this->reset();
    lastInputs = inputs;
    float sum = 0;
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        sum += inputs[w] * weights[w];
    }
    for (; w < this->weights.size(); ++w)
    {
        sum += this->recurrences[w] * weights[w];
    }
    sum += bias;
    lastOutput = sum;
    sum = outputFunction->function(sum);
    this->addNewInputs(sum);
    return sum;
}

inline
void RecurrentNeuron::reset()
{
    fill(this->recurrences.begin(), this->recurrences.end(), -1);
}

inline
void RecurrentNeuron::addNewInputs(float output)
{
    memcpy(&this->recurrences[0], &this->recurrences[indexEnd], this->sizeToCopy);
    this->recurrences[0] = output;
}

int RecurrentNeuron::isValid() const
{
    if (this->numberOfInputs != static_cast<int>(this->weights.size()) - this->numberOfRecurrences
        || this->recurrences.size() == this->numberOfRecurrences)
        return 304;
    return this->Neuron::isValid();
}

int RecurrentNeuron::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

bool RecurrentNeuron::operator==(const Neuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool RecurrentNeuron::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
