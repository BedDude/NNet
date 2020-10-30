using System;
using System.IO;
using System.Collections.Generic;

using NNet.Neurons;
using NNet.Functions;

namespace NNet.IO
{
    /// <summary>
    /// A network configuration serializer.
    /// Saves the network as configuration file (.ncfg) and weights files (.wgt).
    /// </summary>
    public class ConfigSerializer : ISerializer
    {
        /// <summary>
        /// List of the layer - weights file pairs.
        /// </summary>
        private Dictionary<INeuronsLayer, FileStream> _filesTable;


        /// <summary>
        /// Initializes a new instance of <see cref="ConfigSerializer"/>.
        /// </summary>
        public ConfigSerializer()
        {
            _filesTable = new Dictionary<INeuronsLayer, FileStream>();
        }

        /// <summary>
        /// Creates a configuration file and the weights file.
        /// </summary>
        /// <param name="network">The network, which will be saved.</param>
        /// <param name="pathToDir">The directory where the configuration file and directory with the weights files will be created.</param>
        /// <param name="name">Name of the future .ncfg file without extension.</param>
        public void CreateNetworkFile(INeuralNetwork network, string pathToDir, string name)
        {
            var basis = Path.Join(pathToDir, name);
            var file = basis + ".ncfg";
            var dir = basis + "_weights";
            var dirName = name + "_weights";

            Directory.CreateDirectory(dir);
            using (var writer = new StreamWriter(file))
            {
                writer.WriteLine($"path_to_weights = {dirName}");
                writer.WriteLine($"input_size = {network.InputSize}");
                writer.WriteLine($"layers_count = {network.LayersCount}");

                for (int i = 0; i < network.LayersCount; i++)
                {
                    var weightsFile = $"{dirName}_{i}.wgt";
                    var currentLayer = network.Layers[i];

                    writer.WriteLine();
                    writer.WriteLine($"layer_size = {currentLayer.NeuronsCount}");
                    writer.WriteLine($"input_size = {currentLayer.InputSize}");
                    writer.WriteLine($"activation_function = {currentLayer.ActivationFunction}");
                    writer.WriteLine($"weights_file = {weightsFile}");

                    using (var weightsFileStream = new FileStream(Path.Join(dir, weightsFile), FileMode.Create))
                    {
                        _filesTable.Add(currentLayer, weightsFileStream);
                        WriteLayerWeights(currentLayer, weightsFileStream);
                    }
                }
            }
        }

        /// <summary>
        /// Reads a information about network from config and weights files.
        /// </summary>
        /// <param name="file">A .ncfg file.</param>
        /// <returns>A input size - layers pair.</returns>
        public (int, List<INeuronsLayer>) ReadNetworkFile(string file)
        {
            (int, List<INeuronsLayer>) result;

            var localStorage = new Dictionary<string, string>();
            using (var reader = new StreamReader(file))
            {
                for (int i = 0; i < 3; i++)
                {
                    var current = reader.ReadLine()
                                        .Replace(" ", "")
                                        .Split('=', StringSplitOptions.RemoveEmptyEntries);

                    localStorage.Add(current[0], current[1]);
                }

                result.Item1 = int.Parse(localStorage["input_size"]);
                result.Item2 = new List<INeuronsLayer>();

                var layersCount = int.Parse(localStorage["layers_count"]);

                var basePath = Path.Join(file, @"../", localStorage["path_to_weights"]);
                for (int i = 0; i < layersCount; i++)
                {
                    localStorage.Clear();

                    reader.ReadLine();
                    for (int j = 0; j < 4; j++)
                    {
                        var current = reader.ReadLine()
                                            .Replace(" ", "")
                                            .Split('=', StringSplitOptions.RemoveEmptyEntries);

                        localStorage.Add(current[0], current[1]);
                    }
                    var layerSize = int.Parse(localStorage["layer_size"]);
                    var inputSize = int.Parse(localStorage["input_size"]);
                    var functionType = GetFunctionType(localStorage["activation_function"]);
                    var weightsFileStream = new FileStream(Path.Combine(basePath, localStorage["weights_file"]), FileMode.Open);

                    var newLayer = new NeuronsLayer(layerSize, inputSize);
                    newLayer.ActivationFunction = functionType;
                    ReadLayerWeights(newLayer, weightsFileStream);
                    result.Item2.Add(newLayer);

                    _filesTable.Add(newLayer, weightsFileStream);
                }
            }

            return result;

            ActivationFunctionType GetFunctionType(string @string)
            {
                return @string switch
                {
                    "Linear" => ActivationFunctionType.Linear,
                    "Sigmoid" => ActivationFunctionType.Sigmoid,
                    "LReLU" => ActivationFunctionType.LReLU,
                    "ReLU" => ActivationFunctionType.ReLU,
                    "TanH" => ActivationFunctionType.TanH,
                    _ => throw new ArgumentException()
                };
            }
        }

        /// <summary>
        /// Writes all weights of the network to the files.
        /// </summary>
        public void WriteWeights()
        {
            foreach (var layer in _filesTable.Keys)
            {
                WriteLayerWeights(layer, _filesTable[layer]);
            }
        }

        /// <summary>
        /// Reads all layer's weights from the weights file.
        /// </summary>
        /// <param name="layer">The layer of tne network.</param>
        /// <param name="file">A .wgt file.</param>
        private void ReadLayerWeights(INeuronsLayer layer, FileStream file)
        {
            using (var reader = new BinaryReader(file))
            {
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    layer.Bias[i] = reader.ReadDouble();
                    for(int j = 0; j < layer.InputSize; j++)
                    {
                        layer.Weights[j, i] = reader.ReadDouble();
                    }
                }
            }
        }

        /// <summary>
        /// Writes all layer's weights to the weights file.
        /// </summary>
        /// <param name="layer">The layer of tne network.</param>
        /// <param name="file">A .wgt file.</param>
        private void WriteLayerWeights(INeuronsLayer layer, FileStream file)
        {
            using (var writer = new BinaryWriter(file))
            {
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    writer.Write(layer.Bias[i]);
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        writer.Write(layer.Weights[j, i]);
                    }
                }
            }
        }
    }
}