using System.Collections.Generic;

using NNet.IO;
using NNet.Functions;
using NNet.Neurons;

namespace NNet.Builder
{
    /// <summary>
    /// The builder of the neural networks.
    /// </summary>
    public class NeuralNetworkBuilder
    {
        /// <summary>
        /// The result of a network building.
        /// </summary>
        private BuilderResult _result;

        /// <summary>
        /// List of size - function type pairs.
        /// </summary>
        /// <remarks>
        /// It is used to creating network layers.
        /// </remarks>
        private List<(int, ActivationFunctionType)> _pairs;


        /// <summary>
        /// Initializes a new instance of <see cref="NeuralNetworkBuilder"/>.
        /// </summary>
        public NeuralNetworkBuilder()
        {
            _result = new BuilderResult();
            _pairs = new List<(int, ActivationFunctionType)>();
        }

        /// <summary>
        /// Builds a network with the set parameters.
        /// </summary>
        /// <returns>A instance of <see cref="BuilderResult"/> with the set parameters.</returns>
        public BuilderResult Build()
        {
            for (int i = 0; i < _pairs.Count; i++)
            {
                var inputSize = i == 0 ? _result.InputSize : _result.Layers[i - 1].NeuronsCount;

                var newLayer = new NeuronsLayer(_pairs[i].Item1, inputSize);
                newLayer.ActivationFunction = _pairs[i].Item2;

                _result.Layers.Add(newLayer);
            }
            
            return _result;
        }

        /// <summary>
        /// Builds a network from configuration file.
        /// </summary>
        /// <param name="file">A configuration file.</param>
        /// <returns>A instance of <see cref="BuilderResult"/> with the readed parameters.</returns>
        public BuilderResult BuildFromConfigFile(string file)
        {
            _result.Serializer = new ConfigSerializer();
            var configResult = _result.Serializer.ReadNetworkFile(file);

            _result.InputSize = configResult.Item1;
            _result.Layers = configResult.Item2;

            return _result;
        }

        /// <summary>
        /// Builds a network from network file.
        /// </summary>
        /// <param name="file">A network file.</param>
        /// <returns>A instance of <see cref="BuilderResult"/> with the readed parameters.</returns>
        public BuilderResult BuildFromNetworkFile(string file)
        {
            _result.Serializer = new NetworkSerializer();
            var network = _result.Serializer.ReadNetworkFile(file);

            _result.InputSize = network.Item1;
            _result.Layers = network.Item2;

            return _result;
        }

        /// <summary>
        /// Sets a size of input data to the future network.
        /// </summary>
        /// <param name="size">Size of input data.</param>
        /// <returns>This instance of <see cref="NeuralNetworkBuilder"/>.</returns>
        public NeuralNetworkBuilder SetInputSize(int size)
        {
            _result.InputSize = size;

            return this;
        }

        /// <summary>
        /// Sets serializer to the future network.
        /// </summary>
        /// <param name="type">Type of serializer.</param>
        /// <returns>This instance of <see cref="NeuralNetworkBuilder"/>.</returns>
        public NeuralNetworkBuilder SetSerializer(SerializerType type)
        {
            _result.Serializer = type switch
            {
                SerializerType.None => null,
                SerializerType.ConfigSerializer => new ConfigSerializer(),
                SerializerType.NetworkSerializer => new NetworkSerializer()
            };

            return this;
        }

        /// <summary>
        /// Adds new layer to the future network.
        /// </summary>
        /// <param name="neuronsCount">A count of the neurins in the layer.</param>
        /// <param name="type">Type of activation function.</param>
        /// <returns>This instance of <see cref="NeuralNetworkBuilder"/>.</returns>
        public NeuralNetworkBuilder AddLayer(int neuronsCount, ActivationFunctionType type)
        {
            _pairs.Add((neuronsCount, type));

            return this;
        }
    }
}