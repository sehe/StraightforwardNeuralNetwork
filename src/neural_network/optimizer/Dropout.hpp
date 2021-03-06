#pragma once
#include <vector>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "LayerOptimizer.hpp"

namespace snn::internal
{
    class Dropout final : public LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float value;
        float reverseValue;

    public:
        Dropout() = default;  // use restricted to Boost library only
        Dropout(float value);
        Dropout(const Dropout& dropout) = default;
        ~Dropout() = default;

        std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const override;

        void applyBefore(std::vector<float>& inputs) override;
        void applyAfterForBackpropagation(std::vector<float>& outputs) override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
    };

    template <class Archive>
    void Dropout::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<Dropout, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
        ar & this->reverseValue;
    }
}
