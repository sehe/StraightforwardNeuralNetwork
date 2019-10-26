#include <ctime>
#pragma warning(push, 0)
#include <boost/serialization/vector.hpp>
#pragma warning(pop)
#include "neuralNetwork.h"
#include "layer/alltoall.h"
#include <iostream>
#include <omp.h>

using namespace std;
using namespace snn;
using namespace internal;

bool NeuralNetwork::isTheFirst = true;

void NeuralNetwork::initialize()
{
	srand(static_cast<int>(time(nullptr)));
	rand();
	ActivationFunction::initialize();

	//const auto numberOfCore = omp_get_num_procs();
	omp_set_num_threads(12);

	isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structureOfNetwork,
                             const std::vector<activationFunctionType>& activationFunctionByLayer,
                             float learningRate,
                             float momentum,
						     bool useMultithreading) : StatisticAnalysis(structureOfNetwork.back())
{
	if (isTheFirst)
		this->initialize();

	this->structureOfNetwork = structureOfNetwork;
	this->activationFunctionByLayer = activationFunctionByLayer;
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->useMultithreading = useMultithreading;


	this->numberOfLayers = static_cast<int>(structureOfNetwork.size()) - 1;
	this->numberOfHiddenLayers = static_cast<int>(structureOfNetwork.size()) - 2;
	this->numberOfInput = structureOfNetwork[0];
	this->numberOfOutputs = structureOfNetwork.back();

	layers.reserve(numberOfLayers);
	for (unsigned int l = 1; l < structureOfNetwork.size(); ++l)
	{
		Layer* layer(new AllToAll(this->structureOfNetwork[l - 1],
		                          this->structureOfNetwork[l],
		                          this->activationFunctionByLayer[l - 1],
		                          this->learningRate,
		                          this->momentum,
								  this->useMultithreading));
		layers.push_back(layer);
	}

	auto err = this->isValid();
	if (err != 0)
	{
		auto message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of neural networks";
		throw runtime_error(message);
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
	: StatisticAnalysis(neuralNetwork.getNumberOfOutputs())
{
	this->operator=(neuralNetwork);
}

int NeuralNetwork::isValid() const
{
	//TODO: rework isValid
	if (this->numberOfInput < 1 
	|| this->numberOfInput > 2073600) // 1920 * 1080
		return 101;

	if (this->numberOfLayers < 1 
	|| this->numberOfHiddenLayers > 1000
	|| this->numberOfLayers != this->layers.size()
	|| this->numberOfLayers != numberOfHiddenLayers + 1)
		return 102;

	if (learningRate < 0 || learningRate > 1)
		return 103;

	if (momentum < 0 || momentum > 1)
		return 104;

	for (auto& layer : this->layers)
	{
		auto err = layer->isValid();
		if(err != 0)
			return err;
	}
	return 0;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& neuralNetwork)
{
	this->maxOutputIndex = neuralNetwork.maxOutputIndex;
	this->learningRate = neuralNetwork.learningRate;
	this->momentum = neuralNetwork.momentum;
	this->useMultithreading = neuralNetwork.useMultithreading;
	this->numberOfHiddenLayers = neuralNetwork.numberOfHiddenLayers;
	this->numberOfLayers = neuralNetwork.numberOfLayers;
	this->numberOfInput = neuralNetwork.numberOfInput;
	this->numberOfOutputs = neuralNetwork.numberOfOutputs;
	this->structureOfNetwork = neuralNetwork.structureOfNetwork;
	this->activationFunctionByLayer = neuralNetwork.activationFunctionByLayer;

	this->layers.clear();
	this->layers.reserve(neuralNetwork.layers.size());
	for (const auto& layer : neuralNetwork.layers)
	{
		if (layer->getType() == allToAll)
		{
			auto newLayer = new AllToAll();
			newLayer->operator=(*layer);
			this->layers.push_back(newLayer);
		}
		else
			throw exception();
	}

	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	auto equal(this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& this->learningRate == neuralNetwork.learningRate
		&& this->momentum == neuralNetwork.momentum
		&& this->useMultithreading == neuralNetwork.useMultithreading
		&& this->numberOfHiddenLayers == neuralNetwork.numberOfHiddenLayers
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->structureOfNetwork == neuralNetwork.structureOfNetwork
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& this->layers.size() == neuralNetwork.layers.size());

	if (equal)
		for (int l = 0; l < numberOfLayers; l++)
		{
			if (*this->layers[l] != *neuralNetwork.layers[l])
				equal = false;
		}
	return equal;
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
