#include "layer.h"

using namespace std;
using namespace snn;
using namespace internal;

Layer::Layer(const int numberOfInputs,
             const int numberOfNeurons,
             const float learningRate,
             const float momentum,
             const bool useMultithreading)
{
	this->numberOfInputs = numberOfInputs;
	this->numberOfNeurons = numberOfNeurons;
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->useMultithreading = useMultithreading;
}

int Layer::isValid() const
{
	if(this->neurons.size() != this->numberOfNeurons
		|| this->numberOfNeurons < 1
		|| this->numberOfNeurons > 1000000)
		return 201;

	for (auto& neuron : this->neurons)
	{
		auto err = neuron.isValid();
		if(err != 0)
			return err;
	}
	return 0;
}

Perceptron* Layer::getNeuron(int neuronNumber)
{
	return &this->neurons[neuronNumber];
}

Layer& Layer::operator=(const Layer& layer)
{
	this->numberOfInputs = layer.numberOfInputs;
	this->numberOfNeurons = layer.numberOfNeurons;
	this->errors = layer.errors;
	this->neurons = layer.neurons;
	this->learningRate = layer.learningRate;
	this->momentum = layer.momentum;
	this->useMultithreading = layer.useMultithreading;
	return *this;
}

bool Layer::operator==(const Layer& layer) const
{
	return this->numberOfInputs == layer.numberOfInputs
		&& this->numberOfNeurons == layer.numberOfNeurons
		&& this->errors == layer.errors
		&& this->neurons == layer.neurons
		&& this->learningRate == layer.learningRate
		&& this->momentum == layer.momentum
		&& this->useMultithreading == layer.useMultithreading;
}

bool Layer::operator!=(const Layer& layer) const
{
	return !this->operator==(layer);
}
