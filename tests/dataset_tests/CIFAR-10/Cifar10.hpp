#pragma once
#include "../Dataset.hpp"
#include "tools/Tools.hpp"

class Cifar10 final : public Dataset
{
public:
    Cifar10(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
    snn::vector2D<float> readImages(std::string filePaths[], int size, snn::vector2D<float>& labels);
    static void readImages(std::string filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels);
};