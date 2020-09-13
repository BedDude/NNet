using NNet.IO;
using NNet.Neurons;

namespace NNet.Builder
{
    public class SerializableBuilderResult : INeuralNetwork, ISerializable
    {
        public int LayersCount { get; set; }
        public INeuronsLayer[] Layers { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public ISerializer Serializer { get; set; }

        public void CreateConfig(string fileName) { }
        public double[] GetResult(double[] input) => throw new System.NotImplementedException();
        public void Learn(double[] error, double rate) { }
        public void SaveWeights() { }
    }
}