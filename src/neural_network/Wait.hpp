#pragma once
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace snn
{
    enum waitOperator 
    {
        noneOp = 0,
        andOp = 1,
        orOp = 2
    };

    struct Wait
    {
        int epochs = -1;
        float accuracy = -1;
        int duration = -1;
        waitOperator op;
        Wait& operator||(const Wait& wait);
        Wait& operator&&(const Wait& wait);
        bool isOver(int epochs, float accuracy, int duration);
    };

    extern Wait operator""_ep (unsigned long long value);
    extern Wait operator""_acc (long double value);
    extern Wait operator""_ms (unsigned long long value);
    extern Wait operator""_s (unsigned long long value);
    extern Wait operator""_min (unsigned long long value);
}