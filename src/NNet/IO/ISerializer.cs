using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    public interface ISerializer
    {
        string ConfigFileName { get; set; }
        Dictionary<INeuronsLayer, string> FilesTable { get; set; }

        double[,] ReadWeights();
        void WriteWeights();
        INeuronsLayer[] ReadConfig();
        void WriteConfig();
    }
}