#include "alltoall.h"
#pragma warning(push, 0)
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <thread>
#include <algorithm>
#include <execution>
#pragma warning(pop)

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(AllToAll);

AllToAll::AllToAll(const int numberOfInputs,
                   const int numberOfNeurons,
                   activationFunctionType function,
                   float learningRate,
                   float momentum,
                   bool useMultithreading)
	: Layer(numberOfInputs, numberOfNeurons, learningRate, momentum, useMultithreading)
{
	this->neurons.reserve(numberOfNeurons);

	for (int n = 0; n < numberOfNeurons; ++n)
	{
		this->neurons.emplace_back(numberOfInputs, function, learningRate, momentum);
	}
}

vector<float> AllToAll::output(const vector<float>& inputs)
{
	vector<float> outputs(this->numberOfNeurons); // copy in heap or save in RAM, which is faster ?
	int index = 0;
	for_each(execution::par_unseq,
	         neurons.begin(),
	         neurons.end(),
	         [&](auto&& neuron)
	         {
		         outputs[index] = neuron.output(inputs);
		         index = index + 1;
	         });
	return outputs;
}

vector<float> AllToAll::backOutput(vector<float>& inputsError)
{
	vector<float> errors(this->numberOfInputs);

	for (int n = 0; n < numberOfInputs; ++n)
	{
		errors[n] = 0;
	}

	int index = 0;
	for_each(execution::par_unseq,
	         neurons.begin(),
	         neurons.end(),
	         [&](auto&& neuron)
	         {
		         auto result = neuron.backOutput(inputsError[index]);
		         for (int r = 0; r < numberOfInputs; ++r)
			         errors[r] += result[r];
		         index = index + 1;
	         });
	return errors;
}

void AllToAll::train(vector<float>& inputsError)
{
	int index = 0;
	for_each(execution::par_unseq,
	         neurons.begin(),
	         neurons.end(),
	         [&](auto&& neuron)
	         {
		         neuron.backOutput(inputsError[index]);
		         index = index+1;
	         });
}

int AllToAll::isValid() const
{
	return this->Layer::isValid();
}

LayerType AllToAll::getType() const
{
	return allToAll;
}

Layer& AllToAll::operator=(const Layer& layer)
{
	return this->Layer::operator=(layer);
}

bool AllToAll::operator==(const AllToAll& layer) const
{
	return this->Layer::operator==(layer);
}

bool AllToAll::operator!=(const AllToAll& layer) const
{
	return this->Layer::operator!=(layer);
}
