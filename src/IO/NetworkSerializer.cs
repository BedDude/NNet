using System;
using System.IO;
using System.Collections.Generic;

using NNet.Neurons;
using NNet.Functions;

namespace NNet.IO
{
    /// <summary>
    /// A network serializer.
    /// Saves the network as union file of the network (.unf).
    /// </summary>
    public class NetworkSerializer : ISerializer
    {
        /// <summary>
        /// The .unf file.
        /// </summary>
        private string _file;

        /// <summary>
        /// The size of the input data of the network.
        /// </summary>
        private int _inputSize;

        /// <summary>
        /// The layers of the network.
        /// </summary>
        private List<INeuronsLayer> _layers;


        /// <summary>
        /// Initializes a new instance of <see cref="NetworkSerializer"/>.
        /// </summary>
        public NetworkSerializer()
        {
            _file = "";
            _inputSize = 0;
            _layers = new List<INeuronsLayer>();
        }

        /// <summary>
        /// Creates a .unf file with information about the network.
        /// </summary>
        /// <param name="network">The network, which will be saved.</param>
        /// <param name="pathToDir">The directory where the network file will be created.</param>
        /// <param name="name">Name of the future .unf file without extension.</param>
        public void CreateNetworkFile(INeuralNetwork network, string pathToDir, string name)
        {
            _inputSize = network.InputSize;
            _layers = new List<INeuronsLayer>(network.Layers);
            _file = Path.Join(pathToDir, name + ".unf");

            WriteFile();
        }

        /// <summary>
        /// Reads information about a network from the network file.
        /// </summary>
        /// <param name="file">The .unf file.</param>
        /// <returns>A input size - layers pair.</returns>
        public (int, List<INeuronsLayer>) ReadNetworkFile(string file)
        {
            _file = file;

            var localStorage = new Dictionary<string, string>();
            string[] info;
            using(var reader = new StreamReader(_file))
            {
                info = reader.ReadLine().Split(',', StringSplitOptions.RemoveEmptyEntries);
                ConvertStringsToPairs(info);

                _inputSize = int.Parse(localStorage["is"]);
                _layers = new List<INeuronsLayer>();

                var layersCount = int.Parse(localStorage["lc"]);
                for(int i = 0; i < layersCount; i++)
                {
                    localStorage.Clear();
                    reader.ReadLine();

                    info = reader.ReadLine().Split(',', StringSplitOptions.RemoveEmptyEntries);
                    ConvertStringsToPairs(info);

                    var inputSize = int.Parse(localStorage["is"]);
                    var size = int.Parse(localStorage["ls"]);
                    var activationFunction = GetFunctionType(localStorage["af"]);

                    var newLayer = new NeuronsLayer(size, inputSize);
                    newLayer.ActivationFunction = activationFunction;

                    for(int j = 0; j < size; j++)
                    {
                        var weights = reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries);

                        newLayer.Bias[j] = double.Parse(weights[0]);
                        for(int k = 0; k < inputSize; k++)
                        {
                            newLayer.Weights[k, j] = double.Parse(weights[k + 1]);
                        }
                    }

                    _layers.Add(newLayer);
                }
            }

            return (_inputSize, _layers);

            void ConvertStringsToPairs(string[] strings)
            {
                foreach (var param in strings)
                {
                    var pair = param.Split('=', StringSplitOptions.RemoveEmptyEntries);
                    localStorage.Add(pair[0],pair[1]);
                }
            }

            ActivationFunctionType GetFunctionType(string name)
            {
                return name switch
                {
                    "LI" => ActivationFunctionType.Linear,
                    "S" => ActivationFunctionType.Sigmoid,
                    "T" => ActivationFunctionType.TanH,
                    "R" => ActivationFunctionType.ReLU,
                    "LR" => ActivationFunctionType.LReLU
                };
            }
        }

        /// <summary>
        /// Writes all weights of the network.
        /// </summary>
        public void WriteWeights() => WriteFile();

        /// <summary>
        /// Writes all information to the network file.
        /// </summary>
        private void WriteFile()
        {
            using(var writer = new StreamWriter(_file))
            {
                writer.WriteLine($"is={_inputSize},lc={_layers.Count}");

                for(int i = 0; i < _layers.Count; i++)
                {
                    writer.WriteLine();

                    writer.WriteLine($"ls={_layers[i].NeuronsCount},is={_layers[i].InputSize}" + 
                                     $",af={GetActivationFunctionCode(_layers[i].ActivationFunction)}");

                    for(int j = 0; j < _layers[i].NeuronsCount; j++)
                    {
                        writer.Write($"{_layers[i].Bias[j]} ");
                        for(int k = 0; k < _layers[i].InputSize; k++)
                        {
                            writer.Write($"{_layers[i].Weights[k, j]} ");
                        }
                        writer.WriteLine();
                    }
                }
            }

            string GetActivationFunctionCode(ActivationFunctionType type)
            {
                return type switch
                {
                    ActivationFunctionType.Linear => "LI",
                    ActivationFunctionType.Sigmoid => "S",
                    ActivationFunctionType.TanH => "T",
                    ActivationFunctionType.ReLU => "R",
                    ActivationFunctionType.LReLU => "LR"
                };
            }
        }
    }
}