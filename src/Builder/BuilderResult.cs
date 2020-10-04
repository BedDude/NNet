using System.Collections.Generic;

using NNet.IO;
using NNet.Neurons;

namespace NNet.Builder
{
    /// <summary>
    /// Intermediate result of the network construction.
    /// </summary>
    /// <remarks>
    /// It is used to successfully construct a <see cref="NeuralNetwork"/> instance.
    /// </remarks>
    public class BuilderResult
    {
        /// <summary>
        /// Input size of the created network.
        /// </summary>
        public int InputSize { get; set; }

        /// <summary>
        /// Layers of the created network.
        /// </summary>
        public List<INeuronsLayer> Layers { get; set; }

        /// <summary>
        /// Serializer of the created network.
        /// </summary>
        public ISerializer Serializer { get; set; }


        /// <summary>
        /// Initializes a new instance of <see cref="BuilderResult"/>.
        /// </summary>
        public BuilderResult()
        {
            Layers = new List<INeuronsLayer>();
            Serializer = null;
        }

        /// <summary>
        /// Convert this instance to the neural network.
        /// </summary>
        /// <returns>This instance as an instance of <see cref="NeuralNetwork"/>.</returns>
        public NeuralNetwork ToNetwork() => this;
    }
}