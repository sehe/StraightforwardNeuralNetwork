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

	this->sizeOfData = static_cast<int>(trainingInputs.back().size());
	this->numberOfLabel = static_cast<int>(trainingLabels.back().size());;
	this->sets[training].size = static_cast<int>(trainingLabels.size());
	this->sets[testing].size = static_cast<int>(testingLabels.size());

	this->normalization(-1, 1);
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


void Data::normalization(const float min, const float max)
{
	try
	{
		vector<vector<float>>* inputsTraining = &this->sets[training].inputs;
		vector<vector<float>>* inputsTesting = &this->sets[testing].inputs;

		for (auto j = 0; j < this->sizeOfData; j++)
		{
			float minValueOfVector = (*inputsTraining)[0][j];
			float maxValueOfVector = (*inputsTraining)[0][j];

			for (int i = 1; i < (*inputsTraining).size(); i++)
			{
				if ((*inputsTraining)[i][j] < minValueOfVector)
				{
					minValueOfVector = (*inputsTraining)[i][j];
				}
				else if ((*inputsTraining)[i][j] > maxValueOfVector)
				{
					maxValueOfVector = (*inputsTraining)[i][j];
				}
			}

			for (auto i = 0; i < (*inputsTraining).size(); i++)
			{
				(*inputsTraining)[i][j] = ((*inputsTraining)[i][j] - minValueOfVector) / (maxValueOfVector -
					minValueOfVector
				);
				(*inputsTraining)[i][j] = (*inputsTraining)[i][j] * (max - min) + min;
			}
			for (auto i = 0; i < (*inputsTesting).size(); i++)
			{
				(*inputsTesting)[i][j] = ((*inputsTesting)[i][j] - minValueOfVector) / (maxValueOfVector -
					minValueOfVector);
				(*inputsTesting)[i][j] = (*inputsTesting)[i][j] * (max - min) + min;
			}
		}
	}
	catch (std::exception e)
	{
		throw std::exception("Normalization of input data failed");
	}
}

void Data::shuffle()
{
	if (indexes.empty())
	{
		indexes.resize(sets[training].size);
		for (int i = 0; i < static_cast<int>(indexes.size()); i++)
			indexes[i] = i;
	}

	random_device rd;
	mt19937 g(rd());
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

vector<float>& Data::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
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
