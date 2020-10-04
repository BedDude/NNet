namespace NNet.IO
{
    /// <summary>
    /// Provides means to saving the network.
    /// </summary>
    public interface ISerializable
    {
        /// <summary>
        /// A network serializer.
        /// </summary>
        ISerializer Serializer { get; }
        

        /// <summary>
        /// Saves the network as file.
        /// </summary>
        /// <param name="pathToDir"></param>
        /// <param name="name">The name of network.</param>
        void SaveNetwork(string pathToDir, string name);

        /// <summary>
        /// Saves weights of the network.
        /// </summary>
        void SaveWeights();
    }
}