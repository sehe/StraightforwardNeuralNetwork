#include <typeinfo>
#include <boost/serialization/export.hpp>
#include "NeuralNetworkOptimizer.hpp"

using namespace std;
using namespace snn;

//BOOST_CLASS_EXPORT(NeuralNetworkOptimizer)

bool NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code();
}

bool NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
