#include <cmath>
#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(RecurrentNeuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, optimizer)
{
}

float RecurrentNeuron::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset)
        this->reset();
    this->lastInputs = inputs;
    this->previousSum = this->sum;
    this->previousOutput = this->lastOutput;
    this->sum = 0;
    int w;
    for (w = 0; w < (int)inputs.size(); ++w)
    {
        this->sum += inputs[w] * this->weights[w];
    }
    this->sum += this->previousOutput * this->weights[w];
    this->sum += this->bias;
    float output = outputFunction->function(sum);
    this->lastOutput = output;
    return output;
}

std::vector<float>& RecurrentNeuron::backOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    this->optimizer->updateWeights(*this, error);
    return this->errors;
}

void RecurrentNeuron::train(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->optimizer->updateWeights(*this, error);
}

inline
void RecurrentNeuron::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}

int RecurrentNeuron::isValid() const
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 1)
        return 304;
    return this->Neuron::isValid();
}

bool RecurrentNeuron::operator==(const RecurrentNeuron& neuron) const
{
        return this->Neuron::operator==(neuron)
            && this->lastOutput == neuron.lastOutput
            && this->previousOutput == neuron.previousOutput
            && this->recurrentError == neuron.recurrentError
            && this->previousSum == neuron.previousSum;
}

bool RecurrentNeuron::operator!=(const RecurrentNeuron& neuron) const
{
    return !(*this == neuron);
}
