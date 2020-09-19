using System;
using System.IO;
using System.Collections.Generic;

using NNet.Neurons;
using NNet.Functions;

namespace NNet.IO
{
    public class ConfigSerializer : ISerializer
    {
        public Dictionary<INeuronsLayer, string> FilesTable { get; set; }

        public ConfigSerializer()
        {
            FilesTable = new Dictionary<INeuronsLayer, string>();
        }

        public void CreateConfig(INeuralNetwork network, string pathToDir, string name)
        {
            var basis = Path.Join(pathToDir, name);
            var file = basis + ".ncfg";
            var dir = basis + "_weights";

            Directory.CreateDirectory(dir);
            using (var writer = new StreamWriter(file))
            {
                writer.WriteLine($"path_to_weights = {dir}");
                writer.WriteLine($"input_size = {network.InputSize}");
                writer.WriteLine($"layers_count = {network.LayersCount}");

                for (int i = 0; i < network.LayersCount; i++)
                {
                    var weightsFile = $"{dir}_{i}.wgt";

                    writer.WriteLine();
                    writer.WriteLine($"layer_size = {network.Layers[i].NeuronsCount}");
                    writer.WriteLine($"input_size = {network.Layers[i].InputSize}");
                    writer.WriteLine($"activation_function = {network.Layers[i].ActivationFunction}");
                    writer.WriteLine($"path_to_weights = {weightsFile}");

                    var weightsFileWithPath = Path.Join(dir, weightsFile);

                    FilesTable.Add(network.Layers[i], weightsFileWithPath);
                    WriteWeights(network.Layers[i], weightsFileWithPath);
                }
            }
        }

        public (int, List<INeuronsLayer>) ReadConfig(string file)
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
                    var weightsFile = Path.Combine(basePath, localStorage["weights_file"]);

                    var newLayer = new NeuronsLayer(layerSize, inputSize);
                    newLayer.ActivationFunction = functionType;
                    ReadWeights(newLayer, weightsFile);
                    result.Item2.Add(newLayer);

                    FilesTable.Add(newLayer, weightsFile);
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

        public void ReadWeights(INeuronsLayer layer, string file)
        {
            using (var reader = new StreamReader(file))
            {
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    layer.Bias[i] = double.Parse(reader.ReadLine());
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        layer.Weights[j, i] = double.Parse(reader.ReadLine());
                    }
                }
            }
        }

        public void WriteWeights(INeuronsLayer layer, string file)
        {
            using (var writer = new StreamWriter(file))
            {
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    writer.WriteLine(layer.Bias[i]);
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        writer.WriteLine(layer.Weights[j, i]);
                    }
                }
            }
        }
    }
}