using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    public interface WorkingTitle
    {
        Dictionary<INeuronsLayer, string> FilesTable { get; set; }
        
        double[,] ReadWeights();
        void WriteWeights();
    }
}