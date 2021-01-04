#include <algorithm>
#include <functional>
#include "ProblemComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

ProblemComposite::ProblemComposite(Set sets[2], const int numberOfLabels)
    : numberOfLabels(numberOfLabels)
{ 
    this->sets = sets;
}

int ProblemComposite::isValid()
{
    if (&this->sets[training] == nullptr
        && &this->sets[testing] == nullptr)
        return 402;
    return 0;
}

const std::vector<float>& ProblemComposite::getTrainingOutputs(int index, const int batchSize)
{
    int i = this->sets[training].shuffledIndexes[index];
    if (batchSize <= 1)
        return this->sets[training].labels[i];

    batchedLabels.resize(numberOfLabels);

    i = this->sets[training].shuffledIndexes[index];
    const auto data0 = this->sets[training].labels[i];
    i = this->sets[training].shuffledIndexes[index+1];
    const auto data1 = this->sets[training].labels[i];
    transform(data0.begin(), data0.end(), data1.begin(), batchedLabels.begin(), plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        i = this->sets[training].shuffledIndexes[j];
        const auto data = this->sets[training].labels[i];
        transform(batchedLabels.begin(), batchedLabels.end(), data.begin(), batchedLabels.begin(), std::plus<float>());
    }
    transform(batchedLabels.begin(), batchedLabels.end(), batchedLabels.begin(), bind(divides<float>(), placeholders::_1, batchSize));
    return batchedLabels;
}
