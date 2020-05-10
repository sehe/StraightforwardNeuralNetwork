#include <cmath>
#include "StatisticAnalysis.hpp"

using namespace snn;
using namespace internal;

void StatisticAnalysis::initialize(int numberOfCluster)
{
    clusters.resize(numberOfCluster);
    this->startTesting();
}

void StatisticAnalysis::startTesting()
{
    for (auto& c : clusters)
    {
        c.truePositive = 0;
        c.trueNegative = 0;
        c.falsePositive = 0;
        c.falseNegative = 0;
        c.totalError = 0;
    }
    numberOfDataWellClassified = 0;
    numberOfDataMisclassified = 0;
}

void StatisticAnalysis::stopTesting()
{
    float newGlobalClusteringRate = this->computeGlobalClusteringRate();
    float newWeightedClusteringRate = this->computeWeightedClusteringRate();
    float newF1Score = this->computeF1Score();
    float newMeanAbsoluteError = this->computeMeanAbsoluteError();
    float newRootMeanSquaredError = this->computeRootMeanSquaredError();

    this->globalClusteringRateIsBetterThanPreviously = newGlobalClusteringRate > this->globalClusteringRate;
    if (this->globalClusteringRateIsBetterThanPreviously)
        this->globalClusteringRateMax = newGlobalClusteringRate;

    this->weightedClusteringRateIsBetterThanPreviously = newWeightedClusteringRate > this->weightedClusteringRate;
    if (this->weightedClusteringRateIsBetterThanPreviously)
        this->weightedClusteringRateMax = newWeightedClusteringRate;

    this->f1ScoreIsBetterThanPreviously = newF1Score > this->f1Score;
    if (this->f1ScoreIsBetterThanPreviously)
        this->f1Score = newF1Score;

    this->meanAbsoluteErrorIsBetterThanPreviously = newMeanAbsoluteError > this->meanAbsoluteError;
    if(this->meanAbsoluteErrorIsBetterThanPreviously)
        this->meanAbsoluteErrorMax = newMeanAbsoluteError;

    this->rootMeanSquaredErrorIsBetterThanPreviously = newRootMeanSquaredError > this->rootMeanSquaredError;
    if(this->rootMeanSquaredErrorIsBetterThanPreviously)
        this->rootMeanSquaredErrorMax = newMeanAbsoluteError;

    this->globalClusteringRate = newGlobalClusteringRate;
    this->weightedClusteringRate = newWeightedClusteringRate;
    this->f1Score = newF1Score;
    this->meanAbsoluteError = newMeanAbsoluteError;
    this->rootMeanSquaredError = newRootMeanSquaredError;
}

void StatisticAnalysis::evaluateOnceForRegression(const std::vector<float>& outputs,
                                                  const std::vector<float>& desiredOutputs,
                                                  float precision)
{
    bool classifiedWell = true;
    for (int i = 0; i < clusters.size(); i++)
    {
        if (outputs[i] > desiredOutputs[i] + precision)
        {
            clusters[i].falsePositive ++;
            classifiedWell = false;
        }
        else if (outputs[i] < desiredOutputs[i] - precision)
        {
            clusters[i].falseNegative ++;
            classifiedWell = false;
        }
        else if (outputs[i] >= desiredOutputs[i])
        {
            clusters[i].trueNegative ++;
        }
        else if (outputs[i] <= desiredOutputs[i])
        {
            clusters[i].trueNegative ++;
        }

        clusters[i].totalError += abs(desiredOutputs[i] - outputs[i]);
    }
    if (classifiedWell)
        numberOfDataWellClassified++;
    else
        numberOfDataMisclassified++;
}

void StatisticAnalysis::evaluateOnceForMultipleClassification(const std::vector<float>& outputs,
                                                              const std::vector<float>& desiredOutputs,
                                                              float separator)
{
    bool classifiedWell = true;
    for (int i = 0; i < clusters.size(); i++)
    {
        if (outputs[i] > separator && desiredOutputs[i] > separator)
        {
            clusters[i].truePositive ++;
        }
        else if (outputs[i] <= separator && desiredOutputs[i] <= separator)
        {
            clusters[i].trueNegative ++;
        }
        else if (outputs[i] > separator && desiredOutputs[i] <= separator)
        {
            clusters[i].falsePositive ++;
            classifiedWell = false;
        }
        else if (outputs[i] <= separator && desiredOutputs[i] > separator)
        {
            clusters[i].falseNegative ++;
            classifiedWell = false;
        }

        clusters[i].totalError += abs(desiredOutputs[i] - outputs[i]);
    }
    if (classifiedWell)
        numberOfDataWellClassified++;
    else
        numberOfDataMisclassified++;
}

void StatisticAnalysis::evaluateOnceForClassification(const std::vector<float>& outputs, int classNumber)
{
    float maxOutputValue = -2;
    int maxOutputIndex = -1;

    for (int i = 0; i < clusters.size(); i++)
    {
        if (maxOutputValue < outputs[i])
        {
            maxOutputValue = outputs[i];
            maxOutputIndex = i;
        }
        if (i == classNumber)
        {
            clusters[i].totalError += abs(1 - outputs[i]);

            if (outputs[i] > this->separator)
            {
                clusters[i].truePositive ++;
            }
            else
            {
                clusters[i].falseNegative ++;
            }
        }
        else
        {
            clusters[i].totalError += abs(0 - outputs[i]);

            if (outputs[i] > this->separator)
            {
                clusters[i].falsePositive ++;
            }
            else
            {
                clusters[i].trueNegative ++;
            }
        }
    }
    if (maxOutputIndex == classNumber)
        numberOfDataWellClassified++;
    else
        numberOfDataMisclassified++;
}

float StatisticAnalysis::computeGlobalClusteringRate()
{
    return numberOfDataWellClassified / (numberOfDataWellClassified + numberOfDataMisclassified);
}

float StatisticAnalysis::computeWeightedClusteringRate()
{
    float weightedClusteringRate = 0;
    for (const auto& c : clusters)
    {
        const float numerator = c.truePositive;
        const float denominator = c.truePositive + c.falsePositive;

        if (numerator > 0)
            weightedClusteringRate += numerator / denominator;
    }
    return weightedClusteringRate / clusters.size();
}

float StatisticAnalysis::computeF1Score()
{
    float f1Score = 0;
    for (const auto& c : clusters)
    {
        if (c.truePositive > 0)
        {
            const float precision = c.truePositive / (c.truePositive + c.falsePositive);
            const float recall = c.truePositive / (c.truePositive + c.falseNegative);
            f1Score += (precision * recall) / (precision + recall);
        }
    }
    return 2.0f * f1Score / clusters.size();
}

float StatisticAnalysis::computeMeanAbsoluteError()
{
    float totalError = 0;
    for (const auto& c : clusters)
    {
        totalError += c.totalError;
    }
    return totalError / (numberOfDataWellClassified + numberOfDataMisclassified);
}

float StatisticAnalysis::computeRootMeanSquaredError()
{
    float totalError = 0;
    for (const auto& c : clusters)
    {
        totalError += c.totalError * c.totalError;
    }
    const float meanSquaredError = totalError / (numberOfDataWellClassified + numberOfDataMisclassified);
    return sqrt(meanSquaredError);
}

float StatisticAnalysis::getGlobalClusteringRate() const
{
    return this->globalClusteringRate;
}

float StatisticAnalysis::getWeightedClusteringRate() const
{
    return this->weightedClusteringRate;
}

float StatisticAnalysis::getF1Score() const
{
    return this->f1Score;
}

float StatisticAnalysis::getMeanAbsoluteError() const
{
    return this->meanAbsoluteError;
}

float StatisticAnalysis::getRootMeanSquaredError() const
{
    return this->rootMeanSquaredError;
}

bool StatisticAnalysis::operator==(const StatisticAnalysis& sa) const
{
    return this->clusters == sa.clusters
        && this->numberOfDataWellClassified == sa.numberOfDataWellClassified
        && this->numberOfDataMisclassified == sa.numberOfDataMisclassified;
}

bool StatisticAnalysis::operator!=(const StatisticAnalysis& sa) const
{
    return !(*this == sa);
}
