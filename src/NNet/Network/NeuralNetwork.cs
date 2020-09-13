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
            Serializer.ConfigFileName = fileName;
            Serializer.WriteConfig();
        }

        public double[] GetResult(double[] input)
        {
            _lastInput = input;

            double[] currentInput;
            for(int i = 0; i < LayersCount; i++)
            {
                currentInput = i == 0 ? input : Layers[i - 1].Value;

                Layers[i].Active(currentInput);
            }

            return Layers[LayersCount - 1].Value;
        }

        public void Learn(double[] error, double rate)
        {
            throw new System.NotImplementedException();
        }

        public void SaveWeights()
        {
            Serializer.WriteWeights();
        }

        public static implicit operator NeuralNetwork(NonSerializableBuilderResult builderResult)
        {
            return new NeuralNetwork
            {
                Layers = builderResult.Layers,
                LayersCount = builderResult.LayersCount,
                InputSize = builderResult.InputSize,
                OutputSize = builderResult.OutputSize,
                Serializer = null
            };
        }

        public static implicit operator NeuralNetwork(SerializableBuilderResult builderResult)
        {
            return new NeuralNetwork
            {
                Layers = builderResult.Layers,
                LayersCount = builderResult.LayersCount,
                InputSize = builderResult.InputSize,
                OutputSize = builderResult.OutputSize,
                Serializer = builderResult.Serializer
            };
        }
    }
}