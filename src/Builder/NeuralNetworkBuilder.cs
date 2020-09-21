using System;
using System.Collections.Generic;

using NNet.IO;
using NNet.Functions;
using NNet.Neurons;

namespace NNet.Builder
{
    public class NeuralNetworkBuilder
    {
        private Queue<int> _neurons;
        private Queue<ActivationFunctionType> _functions;
        private BuilderResult _result;

        public NeuralNetworkBuilder()
        {
            _result = new BuilderResult();
            _neurons = new Queue<int>();
            _functions = new Queue<ActivationFunctionType>();
        }

        public BuilderResult Build()
        {
            var layersCount = Math.Min(_neurons.Count, _functions.Count);
            for (int i = 0; i < layersCount; i++)
            {
                var neuronsCount = _neurons.Dequeue();
                var inputSize = i == 0 ? _result.InputSize : _result.Layers[i - 1].NeuronsCount;

                var newLayer = new NeuronsLayer(neuronsCount, inputSize);
                newLayer.ActivationFunction = _functions.Dequeue();

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
            _neurons.Enqueue(neuronsCount);
            _functions.Enqueue(type);

            return this;
        }
    }
}