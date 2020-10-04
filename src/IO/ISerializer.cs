using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    /// <summary>
    /// Provides means to save the network.
    /// </summary>
    public interface ISerializer
    {
        /// <summary>
        /// Saves the network's weights.
        /// </summary>
        void WriteWeights();

        /// <summary>
        /// Reads information about network from file.
        /// </summary>
        /// <param name="file">File with infromation about network.</param>
        /// <returns>A input size - layers pair.</returns>
        (int, List<INeuronsLayer>) ReadNetworkFile(string file);

        /// <summary>
        /// Generates file with information about network.
        /// </summary>
        /// <param name="network">The network, which will be saved.</param>
        /// <param name="pathToDir">The directory where the file will be created.</param>
        /// <param name="name">Name of the future file without extension.</param>
        void CreateNetworkFile(INeuralNetwork network, string pathToDir, string name);
    }
}