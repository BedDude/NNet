using System.Collections.Generic;

using NNet.Neurons;

namespace NNet.IO
{
    public interface ISerializer
    {
        void WriteWeights();
        (int, List<INeuronsLayer>) ReadNetworkFile(string file);
        void CreateNetworkFile(INeuralNetwork network, string pathToDir, string name);
    }
}