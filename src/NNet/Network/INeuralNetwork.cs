using NNet.Neurons;
using NNet.IO;

namespace NNet
{
    //подумать о разделении интерфейсов
    public interface INeuralNetwork
    {
        int LayersCount { get; }
        INeuronsLayer[] Layers { get; }
        WorkingTitle Vs { get; }

        double[] GetResult(double[] input);
        void Learn(double[] error, double rate);
        void SaveWeights();
    }
}