﻿#pragma once
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class RectifiedLinearUnit : public ActivationFunction // WARNING : bad function, if sum < 0 at start, neuron will never learn
    {
    private:

        activationFunction getType() const override { return ReLU; }

    public:
        float function(const float x) const override
        {
            return (x > 0.0f) ? 0.0f : x;
        }

        float derivative(const float x) const override
        {
            return (x > 0.0f) ? 0.0f : 1.0f;
        }
    };
}
