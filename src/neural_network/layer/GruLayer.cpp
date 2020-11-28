#include <boost/serialization/export.hpp>
#include "GruLayer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(GruLayer)

GruLayer::GruLayer(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> GruLayer::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<GruLayer>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}