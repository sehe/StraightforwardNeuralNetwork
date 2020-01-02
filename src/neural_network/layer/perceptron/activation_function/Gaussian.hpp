﻿#pragma once
#include "ActivationFunction.hpp"

namespace snn::internal
{
	class Gaussian : public ActivationFunction
	{
	private :

		virtual activationFunction getType() const { return gaussian; }

	public :

		float function(const float x) const override
		{
			return exp(-pow(x, 2));
		}

		float derivative(const float x) const override
		{
			return -2 * x * exp(-pow(x, 2));
		}
	};
}
