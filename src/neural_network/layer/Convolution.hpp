#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Convolution : public Layer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected :
        int numberOfConvolution;
        int sizeOfConvolutionMatrix;
        std::vector<int> shapeOfInput;

    public:
        Convolution() = default;  // use restricted to Boost library only
        Convolution(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Convolution() = default;
        Convolution(const Convolution&) = default;

        //std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override = 0;
        [[nodiscard]] int isValid() const override;

        bool operator==(const Convolution& layer) const;
        bool operator!=(const Convolution& layer) const;
    };

    template <class Archive>
    void Convolution::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
        ar & this->numberOfConvolution;
        ar & this->sizeOfConvolutionMatrix;
        ar & this->shapeOfInput;
    }
}