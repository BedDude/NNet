using NNet.Neurons;

namespace NNet.Builder
{
    public class NonSerializableBuilderResult : INeuralNetwork
    {
        public int LayersCount { get; set; }
        public INeuronsLayer[] Layers { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }

        public double[] GetResult(double[] input) => throw new System.NotImplementedException();
        public void Learn(double[] error, double rate) { }
    }
}