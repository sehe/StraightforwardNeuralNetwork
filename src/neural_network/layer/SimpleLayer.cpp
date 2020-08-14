#include "SimpleLayer.hpp"
#include "neuron/SimpleNeuron.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

using namespace snn;
using namespace internal;

extern template class SimpleLayer<SimpleNeuron>;
extern template class SimpleLayer<RecurrentNeuron>;
extern template class SimpleLayer<GatedRecurrentUnit>;

BOOST_CLASS_EXPORT(SimpleLayer<SimpleNeuron>)
BOOST_CLASS_EXPORT(SimpleLayer<RecurrentNeuron>)
BOOST_CLASS_EXPORT(SimpleLayer<GatedRecurrentUnit>)

template <>
std::vector<float> SimpleLayer<RecurrentNeuron>::output(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}

template <>
std::vector<float> SimpleLayer<GatedRecurrentUnit>::output(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}