#include "layer.h"

using namespace std;
using namespace snn;
using namespace internal;

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
	return *this;
}

bool Layer::operator==(const Layer& layer) const
{
	return this->numberOfInputs == layer.numberOfInputs
		&& this->numberOfNeurons == layer.numberOfNeurons
		&& this->errors == layer.errors
		&& this->neurons == layer.neurons
		&& this->learningRate == layer.learningRate
		&& this->momentum == layer.momentum;
}

bool Layer::operator!=(const Layer& layer) const
{
	return !this->operator==(layer);
}
