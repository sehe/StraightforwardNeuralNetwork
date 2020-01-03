#include <cmath>
#include <iostream>
#include "Perceptron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Perceptron::~Perceptron()
{
    delete this->outputFunction;
}

Perceptron::Perceptron(const int numberOfInputs,
                       activationFunction activation,
                       float* learningRate,
                       float* momentum)
{
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->numberOfInputs = numberOfInputs;

	this->previousDeltaWeights.resize(numberOfInputs, 0);
	this->lastInputs.resize(numberOfInputs, 0);
	this->errors.resize(numberOfInputs, 0);
	this->lastOutput = 0;

	this->activation = activation;
	this->outputFunction = ActivationFunction::create(this->activation);

	this->weights.resize(numberOfInputs);
	for (auto& w : weights)
	{
		w = randomInitializeWeight();
	}
	this->bias = 1.0f;
}

Perceptron::Perceptron(const Perceptron& perceptron)
{
	this->operator=(perceptron);
}

float Perceptron::randomInitializeWeight() const
{
	const float valueMax = 2.4f / sqrt(static_cast<float>(this->numberOfInputs));
	return Tools::randomBetween(-valueMax, valueMax);
}

float Perceptron::output(const vector<float>& inputs)
{
	lastInputs = inputs;
	float sum = 0;
	for (int w = 0; w < numberOfInputs; ++w)
	{
		sum += inputs[w] * weights[w];
	}
	sum += bias;
	lastOutput = sum;
	sum = outputFunction->function(sum);
	return sum;
}

std::vector<float>& Perceptron::backOutput(float error)
{
	error = error * outputFunction->derivative(lastOutput);

	this->train(lastInputs, error);

	for (int w = 0; w < numberOfInputs; ++w)
	{
		errors[w] = error * weights[w];
	}
	return errors;
}

void Perceptron::train(const std::vector<float>& inputs, const float error)
{
	for (int w = 0; w < numberOfInputs; ++w)
	{
		auto deltaWeights = *this->learningRate * error * inputs[w];
		deltaWeights += *this->momentum * this->previousDeltaWeights[w];
		weights[w] += deltaWeights;
		this->previousDeltaWeights[w] = deltaWeights;
	}
}

void Perceptron::addAWeight()
{
	numberOfInputs++;
	weights.push_back(randomInitializeWeight());
}

int Perceptron::isValid() const
{
	if (this->bias != 1)
		return 301;

	if (this->numberOfInputs != weights.size()
		|| this->numberOfInputs < 1
		|| this->numberOfInputs > 1000000)
	{
		return 302;
	}
	for (auto& weight : weights)
		if (weight < -100000 || weight > 10000)
			return 303;

	return 0;
}

vector<float> Perceptron::getWeights() const
{
	return weights;
}

void Perceptron::setWeights(const vector<float>& weights)
{
	this->weights = weights;
}

float Perceptron::getWeight(const int w) const
{
	return weights[w];
}

void Perceptron::setWeight(const int w, const float weight)
{
	this->weights[w] = weight;
}

float Perceptron::getBias() const
{
	return bias;
}

void Perceptron::setBias(const float bias)
{
	this->bias = bias;
}

int Perceptron::getNumberOfInputs() const
{
	return numberOfInputs;
}

Perceptron& Perceptron::operator=(const Perceptron& perceptron)
{
	this->weights = perceptron.weights;
	this->previousDeltaWeights = perceptron.previousDeltaWeights;
	this->lastInputs = perceptron.lastInputs;
	this->errors = perceptron.errors;
	this->lastOutput = perceptron.lastOutput;
	this->numberOfInputs = perceptron.numberOfInputs;
	this->bias = perceptron.bias;
	this->activation = perceptron.activation;
	this->outputFunction = ActivationFunction::create(perceptron.activation);
	return *this;
}

bool Perceptron::operator==(const Perceptron& perceptron) const
{
	return this->weights == perceptron.weights
		&& this->previousDeltaWeights == perceptron.previousDeltaWeights
		&& this->lastInputs == perceptron.lastInputs
		&& this->errors == perceptron.errors
		&& this->lastOutput == perceptron.lastOutput
		&& this->numberOfInputs == perceptron.numberOfInputs
		&& this->bias == perceptron.bias
		&& this->activation == perceptron.activation
		&& *this->outputFunction == *perceptron.outputFunction;
}

bool Perceptron::operator!=(const Perceptron& perceptron) const
{
	return !this->operator==(perceptron);
}
