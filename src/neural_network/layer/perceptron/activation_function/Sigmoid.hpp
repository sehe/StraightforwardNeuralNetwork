﻿#pragma once
#include <cassert>
#include <cmath>
#include <string>
#include "activationFunction.hpp"

namespace snn::internal
{
	class Sigmoid : public ActivationFunction
	{
	private :

		activationFunctionType getType() const override { return sigmoid; }


	public:

		float function(const float x) const override
		{
			float result = 1.0f / (1.0f + exp(-x));
			if (std::isnan(result))
			{
				if (x > 0)
					return 1;
				return 0;
			}
			return result;
		}

		float derivative(const float x) const override
		{
			float result = exp(-x) / pow((exp(-x) + 1.0f), 2);
			if (std::isnan(result))
				return 0.0f;
			assert(!std::isnan(result));
			return result;
		}
	};
}
