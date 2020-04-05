#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "../Optimizer.hpp"
#include "LayerType.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn {
    struct LayerModel;
}

namespace snn::internal
{
    class Layer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        virtual std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const = 0;
        virtual void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const = 0;

        int numberOfInputs;
        std::vector<float> errors;

    public:
        Layer() = default; // use restricted to Boost library only
        Layer(LayerModel& model, StochasticGradientDescent* optimizer);
        Layer(const Layer&) = default;
        virtual ~Layer() = default;

        virtual std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const = 0;

        static const layerType type;

        [[nodiscard]] int getNumberOfInputs() const;
        [[nodiscard]] int getNumberOfNeurons() const;
        [[nodiscard]] virtual std::vector<int> getShapeOfOutput() const = 0;

        std::vector<Perceptron> neurons;

        std::vector<float> output(const std::vector<float>& inputs);
        std::vector<float> backOutput(std::vector<float>& inputErrors);
        void train(std::vector<float>& inputErrors);

        [[nodiscard]] virtual int isValid() const;
        virtual bool operator==(const Layer& layer) const;
        virtual bool operator!=(const Layer& layer) const;
    };

    template <class Archive>
    void Layer::serialize(Archive& ar, unsigned version)
    {
        ar & this->numberOfInputs;
        ar & this->errors;
        ar & this->neurons;
    }
}
