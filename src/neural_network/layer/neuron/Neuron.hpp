#pragma once
#include <cmath>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "BaseNeuron.hpp"
#include "NeuronModel.hpp"
#include "../../optimizer/StochasticGradientDescent.hpp"
#include "../../optimizer/Adam.hpp"
#include "activation_function/ActivationFunction.hpp"
#include "../../../tools/Tools.hpp"

namespace snn::internal
{
    template <class Derived>
    class  Neuron : public BaseNeuron<Derived>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:

        int numberOfInputs;
        std::vector<float> weights;
        float bias;

        // TODO : find an easy way to move them in optimizer
        std::vector<float> deltaWeights; // only use for the SGD optimizer
        std::vector<float> firstMomentWeights; // only use for the Adam optimizer
        std::vector<float> secondRawMomentWeights; // only use for the Adam optimizer

        std::vector<float> lastInputs;
        std::vector<float> errors;

        float sum = 0;

        activation activationFunction;
        std::shared_ptr<ActivationFunction> outputFunction;

        static float randomInitializeWeight(int numberOfInputs);

    public:
        Neuron() = default; // use restricted to Boost library only
        Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Neuron(const Neuron& neuron) = default;
        ~Neuron() = default;

        [[nodiscard]] int isValid() const;

        [[nodiscard]] std::vector<float> getWeights() const;
        [[nodiscard]] int getNumberOfParameters() const;
        [[nodiscard]] int getNumberOfInputs() const;

        bool operator==(const Neuron& neuron) const;
        bool operator!=(const Neuron& neuron) const;
    };

    template <class Derived>
    template <class Archive>
    void Neuron<Derived>::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<Neuron, BaseNeuron<Derived>>();
        ar & boost::serialization::base_object<BaseNeuron<Derived>>(*this);
        ar & this->numberOfInputs;
        ar & this->weights;
        ar & this->bias;
        ar & this->deltaWeights;
        ar & this->lastInputs;
        ar & this->errors;
        ar & this->sum;
        ar & this->activationFunction;
        this->outputFunction = ActivationFunction::get(activationFunction);
    }

    #include "Neuron.tpp"
}
