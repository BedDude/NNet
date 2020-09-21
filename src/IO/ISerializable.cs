namespace NNet.IO
{
    public interface ISerializable
    {
        ISerializer Serializer { get; }

        void CreateConfig(string pathToDir, string name);
        void SaveWeights();
    }
}