using NNet.Neurons;

namespace NNet
{
    /// <summary>
    /// Provides means to interacting with the network.
    /// </summary>
    public interface INeuralNetwork
    {
        /// <summary>
        /// The count of the layers in the network.
        /// </summary>
        int LayersCount { get; }

        /// <summary>
        /// The layers of the network.
        /// </summary>
        INeuronsLayer[] Layers { get; }

        /// <summary>
        /// The size of the input data.
        /// </summary>
        int InputSize { get; }
        
        /// <summary>
        /// The size pf the output data.
        /// </summary>
        int OutputSize { get; }
        

        /// <summary>
        /// Calculates result of the network's work.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>The result of the calculation.</returns>
        double[] GetResult(double[] input);

        /// <summary>
        /// The one iteration of the network learning.
        /// </summary>
        /// <param name="error">The error of the calculation.</param>
        /// <param name="rate">The rate of the learning.</param>
        void Learn(double[] error, double rate);
    }
}