using System.Collections.Generic;

using NNet.IO;
using NNet.Neurons;

namespace NNet.Builder
{
    public class BuilderResult
    {
        public int InputSize { get; set; }
        public List<INeuronsLayer> Layers { get; set; }
        public ISerializer Serializer { get; set; }

        public BuilderResult()
        {
            Layers = new List<INeuronsLayer>();
            Serializer = null;
        }

        public NeuralNetwork ToNetwork() => this;
    }
}