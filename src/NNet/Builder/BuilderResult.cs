using NNet.IO;
using NNet.Neurons;

namespace NNet.Builder
{
    public class BuilderResult
    {
        public int InputSize { get; set; }
        public INeuronsLayer[] Layers { get; set; }
        public ISerializer Serializer { get; set; }

        public NeuralNetwork ToNetwork() => this;
    }
}