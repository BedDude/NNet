namespace NNet.IO
{
    public interface ISerializable
    {
        ISerializer Serializer { get; }

        void SaveNetwork(string pathToDir, string name);
        void SaveWeights();
    }
}