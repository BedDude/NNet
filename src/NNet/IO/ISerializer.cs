using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    public interface ISerializer
    {
        string ConfigFileName { get; set; }
        Dictionary<INeuronsLayer, string> FilesTable { get; set; }

        double[,] ReadWeights(string file);
        void WriteWeights(string file);
        (int, INeuronsLayer[]) ReadConfig();
        void CreateConfig();
    }
}