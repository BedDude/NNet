using NNet.Neurons;
using NNet.IO;
using NNet.Builder;

namespace NNet
{
    public class NeuralNetwork : INeuralNetwork, ISerializable
    {
        private double[] _lastInput;

        public int LayersCount { get; private set; }
        public INeuronsLayer[] Layers { get; private set; }
        public int InputSize { get; private set; }
        public int OutputSize { get; private set; }
        public ISerializer Serializer { get; private set; }
        public static NeuralNetworkBuilder Builder => new NeuralNetworkBuilder();

        public void CreateConfig(string fileName)
        {
            if (Serializer != null)
            {
                Serializer.ConfigFileName = fileName;
                Serializer.WriteConfig();
            }
        }

        public double[] GetResult(double[] input)
        {
            _lastInput = input;

            double[] currentInput;
            for (int i = 0; i < LayersCount; i++)
            {
                currentInput = i == 0 ? input : Layers[i - 1].Value;

                Layers[i].Active(currentInput);
            }

            return Layers[LayersCount - 1].Value;
        }

        public void Learn(double[] error, double rate)
        {
            Layers[LayersCount - 1].Error = error;
            for (int i = LayersCount - 1; i >= 1; i--)
            {
                Layers[i].TranslateError(Layers[i - 1]);

                Layers[i].Learn(Layers[i - 1].Value, rate);
            }
            Layers[0].Learn(_lastInput, rate);
        }

        public void SaveWeights()
        {
            Serializer?.WriteWeights();
        }

        public static implicit operator NeuralNetwork(BuilderResult result)
        {
            var layersCount = result.Layers.Length;
            var outputSize = result.Layers[layersCount].NeuronsCount;

            return new NeuralNetwork
            {
                Layers = result.Layers,
                LayersCount = layersCount,
                InputSize = result.InputSize,
                OutputSize = outputSize,
                Serializer = result.Serializer
            };
        }
    }
}