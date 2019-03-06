#include "DataForClassification.h"
using namespace std;

DataForClassification::DataForClassification(std::vector<std::vector<float>>& trainingInputs,
                                             std::vector<std::vector<float>>& trainingLabels,
                                             std::vector<std::vector<float>>& testingInputs,
                                             std::vector<std::vector<float>>& testingLabels)
	: Data(trainingInputs, trainingLabels, testingInputs, testingLabels)
{
	this->problem = classification;
}

int DataForClassification::getTrainingLabel(const int index)
{
	for (int i = 0; i < this->numberOfLabel; i++)
	{
		if (this->sets[training].labels[indexes[index]][i] == 1)
			return i;
	}
	throw exception("wrong label");
}

int DataForClassification::getTestingLabel(const int index)
{
	for (int i = 0; i < this->numberOfLabel; i++)
	{
		if (this->sets[testing].labels[index][i] == 1)
			return i;
	}
	throw exception("wrong label");
}

vector<float>& DataForClassification::getTestingOutputs(const int index)
{
	// Should never be called
	throw exception();
}

