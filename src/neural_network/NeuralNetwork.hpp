#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include "NeuralNetworkOption.hpp"
#include "layer/Layer.hpp"
#include "layer/LayerModel.hpp"
#include "layer/perceptron/activation_function/ActivationFunction.hpp"
#include "StatisticAnalysis.hpp"


namespace snn::internal
{
	class NeuralNetwork : public StatisticAnalysis
	{
	private :
		static bool isTheFirst;
		static void initialize();

		int maxOutputIndex{};
		int numberOfHiddenLayers{};
		int numberOfLayers;
		int numberOfInput{};
		int numberOfOutputs{};

		std::vector<int> structureOfNetwork{};
		std::vector<snn::activationFunctionType> activationFunctionByLayer{};

		std::vector<std::unique_ptr<Layer>> layers{};

		void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired);
		std::vector<float>& calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const;

		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, unsigned version);

	protected :
		NeuralNetwork(const std::vector<LayerModel>& models);

		NeuralNetwork(const NeuralNetwork& neuralNetwork);

		NeuralNetwork() = default;
		~NeuralNetwork() = default;

		NeuralNetworkOption* option;

		[[nodiscard]] std::vector<float> output(const std::vector<float>& inputs);

		void evaluateOnceForRegression(const std::vector<float>& inputs,
		                               const std::vector<float>& desired,
		                               float precision);
		void evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
		                                           const std::vector<float>& desired,
		                                           float separator);
		void evaluateOnceForClassification(const std::vector<float>& inputs, int classNumber);

		int getLastError() const;

		NeuralNetwork& operator=(const NeuralNetwork& neuralNetwork);
		bool operator==(const NeuralNetwork& neuralNetwork) const;
		bool operator!=(const NeuralNetwork& neuralNetwork) const;

	public:
		int isValid() const;

		void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired);

		int getNumberOfInputs() const;
		int getNumberOfHiddenLayers() const;
		int getNumberOfNeuronsInLayer(int layerNumber) const;
		Layer& getLayer(int layerNumber);
		activationFunctionType getActivationFunctionInLayer(int layerNumber) const;
		int getNumberOfOutputs() const;
	};

	template <class Archive>
	void NeuralNetwork::serialize(Archive& ar, const unsigned int version)
	{
		boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
		ar & boost::serialization::base_object<StatisticAnalysis>(*this);
		ar & this->option;
		ar & this->maxOutputIndex;
		ar & this->numberOfHiddenLayers;
		ar & this->numberOfLayers;
		ar & this->numberOfInput;
		ar & this->numberOfOutputs;
		ar & this->structureOfNetwork;
		ar & this->activationFunctionByLayer;
		ar & this->numberOfInput;
		ar.template register_type<AllToAll>();
		ar & layers;
	}
}
