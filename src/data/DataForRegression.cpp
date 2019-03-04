#include "DataForRegression.h"
using namespace std;

vector<float>& DataForRegression::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}

DataForRegression::DataForRegression(std::vector<std::vector<float>>& trainingInputs,
                                     std::vector<std::vector<float>>& trainingLabels,
                                     std::vector<std::vector<float>>& testingInputs,
                                     std::vector<std::vector<float>>& testingLabels)
	: Data(trainingInputs, trainingLabels, testingInputs, testingLabels)
{
	this->problem = regression;
}