template <class N>
Layer<N>::Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.neuron, optimizer);
    }
    LayerOptimizerFactory::build(this->optimizers, model);
}

template<class N>
Layer<N>::Layer(const Layer& layer)
{
    this->numberOfInputs = layer.numberOfInputs;
    this->neurons = layer.neurons;

    this->optimizers.reserve(layer.optimizers.size());
    for(auto& optimizer : layer.optimizers)
        this->optimizers.emplace_back(optimizer->clone(optimizer.get()));
}

template <class N>
std::vector<float> Layer<N>::output(const std::vector<float>& inputs, bool temporalReset)
{
    if(this->optimizers.size() > 0)
    {
        auto copyOfInputs = std::vector(inputs);
        for (auto& optimizer : this->optimizers)
            optimizer->applyBefore(copyOfInputs);
        auto output = this->computeOutput(copyOfInputs, temporalReset);
        return output;
    }
    else
    {
        auto output = this->computeOutput(inputs, temporalReset);
        return output;
    }
}

template <class N>
std::vector<float> Layer<N>::outputForBackpropagation(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = this->computeOutput(inputs, temporalReset);
    for(auto& optimizer : this->optimizers)
        optimizer->applyAfterForBackpropagation(output);
    return output;
}

template <class N>
void Layer<N>::train(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].train(inputErrors[n]);
    }
}

template <class N>
int Layer<N>::isValid() const
{
    if (this->getNumberOfNeurons() != (int)this->neurons.size()
        || this->getNumberOfNeurons() < 1
        || this->getNumberOfNeurons() > 1000000)
        return 201;

    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (int s : shape)
        numberOfOutput *= s;

    if (numberOfOutput != this->getNumberOfNeurons())
        return 202;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if (err != 0)
            return err;
    }
    return 0;
}

template <class N>
void* Layer<N>::getNeuron(int index)
{
    return static_cast<void*>(&this->neurons[index]);
}

template <class N>
int Layer<N>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <class N>
int Layer<N>::getNumberOfNeurons() const
{
    return (int)this->neurons.size();
}

template <class N>
int Layer<N>::getNumberOfParameters() const
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

template <class N>
bool Layer<N>::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& l = dynamic_cast<const Layer&>(layer);

        return typeid(*this).hash_code() == typeid(layer).hash_code()
            && this->numberOfInputs == l.numberOfInputs
            && this->neurons == l.neurons
            && [this, &l]()
            {
                for (size_t o = 0; o < this->optimizers.size(); ++o)
                {
                    if (*this->optimizers[o] != *l.optimizers[o])
                        return false;
                }
                return true;
            }();
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template <class N>
bool Layer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
