#include <fstream>
#include "Mnist.hpp"
#include "data/DataForClassification.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Mnist::Mnist()
{
    this->loadData();
}

void Mnist::loadData()
{
    vector2D<float> trainingInputs = this->readImages("./MNIST/train-images.idx3-ubyte", 60000);
    vector2D<float> trainingLabels = this->readLabels("./MNIST/train-labels.idx1-ubyte", 60000);
    vector2D<float> testingInputs = this->readImages("./MNIST/t10k-images.idx3-ubyte", 10000);
    vector2D<float> testingLabels = this->readLabels("./MNIST/t10k-labels.idx1-ubyte", 10000);

    this->data = make_unique<DataForClassification>(trainingInputs, trainingLabels, testingInputs, testingLabels);
}

vector2D<float> Mnist::readImages(string filePath, int size)
{
    ifstream file;
    file.open(filePath, ios::in | ios::binary);
    vector2D<float> images;
    images.reserve(size);
    constexpr int sizeOfData = 28 * 28;

    if (!file.is_open())
        throw FileOpeningFailed();

    unsigned char c;
    int shift = 0;
    for (int i = 0; !file.eof(); i++)
    {
        vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(sizeOfData);
        if (!file.eof())
            for (int j = 0; !file.eof() && j < sizeOfData;)
            {
                c = file.get();

                if (shift > 15)
                {
                    float value = static_cast<int>(c) / 255.0f * 2.0f - 1.0f;
                    images.back().push_back(value);
                    j++;
                }
                else
                    shift ++;
            }
    }
    file.close();
    return images;
}

vector2D<float> Mnist::readLabels(string filePath, int size)
{
    ifstream file;
    file.open(filePath, ios::in | ios::binary);
    vector2D<float> labels;
    labels.reserve(size);

    if (!file.is_open())
        throw FileOpeningFailed();

    unsigned char c;
    for (int i = 0; !file.eof(); i++)
    {
        c = file.get();

        vector<float> labelsTemp(10, 0);
        labels.push_back(labelsTemp);

        if (!file.eof())
            labels.back()[c] = 1.0;
        else
            labels.resize(labels.size() - 1);
    }
    file.close();
    return labels;
}