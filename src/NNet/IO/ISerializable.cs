namespace NNet.IO
{
    public interface ISerializable
    {
        ISerializer Serializer { get; }

        void CreateConfig(string fileName);
        void SaveWeights();
    }
}