namespace NNet.IO
{
    public interface ISerializeable
    {
        ISerializer Serializer { get; }

        void CreateConfig(string fileName);
        void SaveWeights();
    }
}