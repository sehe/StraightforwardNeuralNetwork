#include <cstdlib>
#include <stdexcept>
#include "Tools.hpp"
using namespace std;
using namespace snn;
using namespace internal;


int Tools::randomBetween(const int min, const int max) // [min; max[
{
	return rand() % (max - min) + min;
}

float Tools::randomBetween(const float min, const float max)
{
	return rand() / static_cast<float>(RAND_MAX) * (max - min) + min;
}

template <typename T>
T Tools::getMinValue(vector<T> vector)
{
	if (vector.size() > 1)
	{
		T minValue = vector[0];

		for (int i = 1; i < vector.size(); i++)
		{
			if (vector[i] < minValue)
			{
				minValue = vector[i];
			}
		}
		return minValue;
	}
	throw runtime_error("Vector is empty");
}

template <typename T>
T Tools::getMaxValue(vector<T> vector)
{
	if (vector.size() > 1)
	{
		T maxValue = vector[0];

		for (int i = 1; i < vector.size(); i++)
		{
			if (vector[i] > maxValue)
			{
				maxValue = vector[i];
			}
		}
		return maxValue;
	}
	throw runtime_error("Vector is empty");
}