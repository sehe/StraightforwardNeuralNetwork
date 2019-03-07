#include <algorithm>
#include <vector>
#include "Data.h"
#include <random>
using namespace std;
using namespace snn;

Data::Data(std::vector<std::vector<float>>& trainingInputs,
           std::vector<std::vector<float>>& trainingLabels,
           std::vector<std::vector<float>>& testingInputs,
           std::vector<std::vector<float>>& testingLabels)
{
	this->sets[training].inputs = trainingInputs;
	this->sets[training].labels = trainingLabels;
	this->sets[testing].inputs = testingInputs;
	this->sets[testing].labels = testingLabels;

	this->sizeOfData = trainingInputs.back().size();
	this->numberOfLabel = trainingLabels.back().size();;
	this->sets[training].size = trainingLabels.size();
	this->sets[testing].size = testingLabels.size();
}

void Data::clearData()
{
	this->sets[training].labels.clear();
	this->sets[training].inputs.clear();
	this->sets[testing].labels.clear();
	this->sets[testing].inputs.clear();
	this->sets[training].size = 0;
	this->sets[testing].size = 0;
}

void Data::shuffle()
{
	if (indexes.empty())
	{
		indexes.resize(sets[training].size);
		for (int i = 0; i < static_cast<int>(indexes.size()); i++)
			indexes[i] = i;
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indexes.begin(), indexes.end(), g);
}

void Data::unshuffle()
{
	indexes.resize(sets[training].size);
	for (int i = 0; i < static_cast<int>(indexes.size()); i++)
		indexes[i] = i;
}

vector<float>& Data::getTrainingData(const int index)
{
	return this->sets[training].inputs[indexes[index]];
}

vector<float>& Data::getTestingData(const int index)
{
	return this->sets[testing].inputs[index];
}

vector<float>& Data::getTrainingOutputs(const int index)
{
	return this->sets[training].labels[indexes[index]];
}

std::vector<float>& Data::getData(set set, const int index)
{
	if (set == training)
		return this->getTrainingData(index);

	return this->getTestingData(index);
}

std::vector<float>& Data::getOutputs(set set, const int index)
{
	if (set == training)
		return this->getTrainingOutputs(index);

	return this->getTestingOutputs(index);
}

int Data::getLabel(set set, const int index)
{
	if (set == training)
		return this->getTrainingLabel(index);

	return this->getTestingLabel(index);
}
