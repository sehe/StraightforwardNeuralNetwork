﻿#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Tanh final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::tanh; }

    public:
        Tanh()
            : ActivationFunction(-1, 1)
        {
        }

        float function(const float x) const override
        {
            return std::tanh(x);
        }

        float derivative(const float x) const override
        {
            return 1 - powf(std::tanh(x), 2);
        }
    };
}