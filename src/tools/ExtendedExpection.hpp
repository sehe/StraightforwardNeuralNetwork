#include <stdexcept>

namespace snn
{
    class NotImplementedException : public std::runtime_error
    {
    public:
        NotImplementedException(std::string functionName = "Function") : std::runtime_error(functionName + " not yet implemented.")
        {
        }
    };

    class FileOpeningFailed : public std::runtime_error
    {
    public:
        FileOpeningFailed() : std::runtime_error("Cannot open file.")
        {
        }
    };

    class InvalidArchitectureException : public std::runtime_error
    {
    public:
        InvalidArchitectureException(std::string message) : std::runtime_error("Invalid neural network architecture: " + message)
        {
        }
    };
}