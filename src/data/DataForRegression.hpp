#pragma once
#include "Data.hpp"

namespace snn
{
	class DataForRegression : public Data
	{
	private:
		[[nodiscard]] const std::vector<float>& getTestingOutputs(const int index) override;

	public:
		DataForRegression(std::vector<std::vector<float>> trainingInputs,
		                  std::vector<std::vector<float>> trainingLabels,
		                  std::vector<std::vector<float>> testingInputs,
		                  std::vector<std::vector<float>> testingLabels,
		                  float precision = 0.1f);

		DataForRegression(std::vector<std::vector<float>> inputs,
		                  std::vector<std::vector<float>> labels,
		                  float precision);
	};
}
