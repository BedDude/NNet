using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    public interface ISerializer
    {
        Dictionary<INeuronsLayer, string> FilesTable { get; set; }

        void ReadWeights(INeuronsLayer layer, string file);
        void WriteWeights(INeuronsLayer layer, string file);
        (int, List<INeuronsLayer>) ReadConfig(string file);
        void CreateConfig(INeuralNetwork network, string pathToDir, string name);
    }
}