using NNet.Neurons;

namespace NNet
{
    public interface INeuralNetwork
    {
        int LayersCount { get; }
        INeuronsLayer[] Layers { get; }
        int InputSize { get; }
        int OutputSize { get; }

        double[] GetResult(double[] input);
        void Learn(double[] error, double rate);
    }
}