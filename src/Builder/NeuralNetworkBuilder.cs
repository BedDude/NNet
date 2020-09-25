using System;
using System.Collections.Generic;

using NNet.IO;
using NNet.Functions;
using NNet.Neurons;

namespace NNet.Builder
{
    public class NeuralNetworkBuilder
    {
        private BuilderResult _result;
        private List<(int, ActivationFunctionType)> _pairs;

        public NeuralNetworkBuilder()
        {
            _result = new BuilderResult();
            _pairs = new List<(int, ActivationFunctionType)>();
        }

        public BuilderResult Build()
        {
            for (int i = 0; i < _pairs.Count; i++)
            {
                var inputSize = i == 0 ? _result.InputSize : _result.Layers[i - 1].NeuronsCount;

                var newLayer = new NeuronsLayer(_pairs[i].Item1, inputSize);
                newLayer.ActivationFunction = _pairs[i].Item2;

                _result.Layers.Add(newLayer);
            }
            
            return _result;
        }

        public BuilderResult BuildFromConfig(string file)
        {
            _result.Serializer = new ConfigSerializer();
            var configResult = _result.Serializer.ReadNetworkFile(file);

            _result.InputSize = configResult.Item1;
            _result.Layers = configResult.Item2;

            return _result;
        }

        public NeuralNetworkBuilder SetInputSize(int size)
        {
            _result.InputSize = size;

            return this;
        }

        public NeuralNetworkBuilder SetSerializer(SerializerType type)
        {
            _result.Serializer = type switch
            {
                SerializerType.None => null,
                SerializerType.ConfigSerializer => new ConfigSerializer()
            };

            return this;
        }

        public NeuralNetworkBuilder AddLayer(int neuronsCount, ActivationFunctionType type)
        {
            _pairs.Add((neuronsCount, type));

            return this;
        }
    }
}