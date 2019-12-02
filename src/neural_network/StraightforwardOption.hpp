#pragma once
#include <string>
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#pragma warning(pop)

namespace snn
{
	class StraightforwardOption : NeuralNetworkOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		bool autoSaveWhenBetter = false;
		std::string saveFilePath = "save.snn";

		StraightforwardOption() = default;
		~StraightforwardOption() = default;

		bool operator==(const StraightforwardOption& option) const;
		StraightforwardOption& operator=(const StraightforwardOption& option) = default;
	};

	template <class Archive>
	void StraightforwardOption::serialize(Archive& ar, const unsigned version)
	{
		ar & this->autoSaveWhenBetter;
		ar & this->saveFilePath;
		boost::serialization::void_cast_register<StraightforwardOption, NeuralNetworkOption>();
		ar & boost::serialization::base_object<NeuralNetworkOption>(*this);
	}
}
