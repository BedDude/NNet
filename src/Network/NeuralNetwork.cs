using NNet.Neurons;
using NNet.IO;
using NNet.Builder;

namespace NNet
{
    /// <summary>
    /// A neural network class.
    /// </summary>
    public class NeuralNetwork : INeuralNetwork, ISerializable
    {
        /// <summary>
        /// The last input data.
        /// </summary>
        /// <remarks>
        /// It is uses in the learning of the network.
        /// </remarks>
        private double[] _lastInput;

        /// <summary>
        /// The count of the layers.
        /// </summary>
        public int LayersCount { get; private set; }

        /// <summary>
        /// The layers of the network.
        /// </summary>
        public INeuronsLayer[] Layers { get; private set; }

        /// <summary>
        /// The size of the input data.
        /// </summary>
        public int InputSize { get; private set; }

        /// <summary>
        /// The size of the output data.
        /// </summary>
        public int OutputSize { get; private set; }

        /// <summary>
        /// The serializer of the network.
        /// </summary>
        public ISerializer Serializer { get; private set; }


        /// <summary>
        /// Saves the network as file.
        /// </summary>
        /// <param name="pathToDir">The directory where the file will be created.</param>
        /// <param name="name">The name of network.</param>
        public void SaveNetwork(string pathToDir, string name)
        {
            Serializer.CreateNetworkFile(this, pathToDir, name);
        }

        /// <summary>
        /// Calculates result of the network's work.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>Result of calculation.</returns>
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

        /// <summary>
        /// Learns the network.
        /// </summary>
        /// <param name="error">The error of the calculation.</param>
        /// <param name="rate">The rate of the learning.</param>
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

        /// <summary>
        /// Saves weights of this network.
        /// </summary>
        public void SaveWeights() => Serializer.WriteWeights();


        /// <summary>
        /// Generates a new instance of <see cref="NeuralNetworkBuilder"/>.
        /// </summary>
        /// <returns>A new instance of <see cref="NeuralNetworkBuilder"/>.</returns>
        public static NeuralNetworkBuilder Builder => new NeuralNetworkBuilder();

        /// <summary>
        /// Converts a instance of <see cref="BuilderResult"/> to a instance of <see cref="NeuralNetwork"/>.
        /// </summary>
        /// <param name="result">The instance of <see cref="BuilderResult"/> with some parametrs.</param>
        public static implicit operator NeuralNetwork(BuilderResult result)
        {
            var layersCount = result.Layers.Count;
            var outputSize = layersCount == 0 ? 0 : result.Layers[layersCount - 1].NeuronsCount;

            return new NeuralNetwork
            {
                Layers = result.Layers.ToArray(),
                LayersCount = layersCount,
                InputSize = result.InputSize,
                OutputSize = outputSize,
                Serializer = result.Serializer
            };
        }
    }
}