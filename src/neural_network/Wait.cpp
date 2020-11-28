#include <algorithm>
#include <stdexcept>
#include "Wait.hpp"

using namespace snn;
using namespace std;


Wait& Wait::operator||(const Wait& wait)
{
    if (op == waitOperator::andOp)
        throw runtime_error("Cannot mix || and && operator for waitFor.");

    this->op = waitOperator::orOp;

    if (this->epochs > 0 && wait.epochs > 0)
        this->epochs = min(this->epochs, wait.epochs);
    else
        this->epochs = max(this->epochs, wait.epochs);

    if (this->accuracy > 0 && wait.accuracy > 0)
        this->accuracy = min(this->accuracy, wait.accuracy);
    else
        this->accuracy = max(this->accuracy, wait.accuracy);

    if (this->duration > 0 && wait.duration > 0)
        this->duration = min(this->duration, wait.duration);
    else
        this->duration = max(this->duration, wait.duration);

    return *this;
}

Wait& Wait::operator&&(const Wait& wait)
{
    if (op == waitOperator::orOp)
        throw runtime_error("Cannot mix || and && operator for waitFor.");

    this->op = waitOperator::andOp;

    this->epochs = max(this->epochs, wait.epochs);
    this->accuracy = max(this->accuracy, wait.accuracy);
    this->duration = max(this->duration, wait.duration);

    return *this;
}

bool Wait::isOver(int currentEpochs, float CurrentAccuracy, float currentMae, int currentDuration) const
{
    const bool isValidEpochs = currentEpochs >= this->epochs && currentEpochs > 0;
    const bool isValidAccuracy = CurrentAccuracy >= this->accuracy && CurrentAccuracy > 0;
    const bool isValidMae = currentMae <= this->mae && currentMae > 0;
    const bool isValidDuration = currentDuration >= this->duration;

    if (this->op == waitOperator::andOp)
    {
        if ((isValidEpochs || this->epochs < 0)
            && (isValidAccuracy || this->accuracy < 0)
            && (isValidMae || this->mae < 0)
            && (isValidDuration || this->duration < 0))
            return true;
    }
    else if ((isValidEpochs && this->epochs >= 0)
        || (isValidAccuracy && this->accuracy >= 0)
        || (isValidMae && this->mae >= 0)
        || (isValidDuration && this->duration >= 0))
        return true;
    return false;
}

Wait snn::operator""_ep(unsigned long long value)
{
    Wait res;
    res.epochs = (int)value;
    return res;
}

Wait snn::operator""_acc(long double value)
{
    Wait res;
    res.accuracy = (float)value;
    return res;
}

Wait snn::operator""_mae(long double value)
{
    Wait res;
    res.mae = (float)value;
    return res;
}

Wait snn::operator""_ms(unsigned long long value)
{
    Wait res;
    res.duration = (int)value;
    return res;
}

Wait snn::operator""_s(unsigned long long value)
{
    Wait res;
    res.duration = (int)value * 1000;
    return res;
}

Wait snn::operator""_min(unsigned long long value)
{
    Wait res;
    res.duration = (int)value * 1000 * 60;
    return res;
}