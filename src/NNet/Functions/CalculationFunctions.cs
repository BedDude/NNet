using System;

using NNet.Neurons;

namespace NNet.Functions
{
    //пока internal, а там как пойдет
    internal static class CalculationFunctions
    {
        public static Func<double[], INeuronsLayer, double[]> GetActivationFunction(ActivationFunctionType functionType)
        {
            var function = GetFunction(functionType);

            return (input, layer) =>
            {
                var result = new double[layer.NeuronsCount];

                double sumOfInputs;
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    sumOfInputs = 0;
                    for (int j = 0; j < input.Length; j++)
                    {
                        sumOfInputs += input[i] * layer.Weights[j, i];
                    }
                    sumOfInputs += layer.Bias[i];

                    result[i] = function(sumOfInputs);
                }

                return result;
            };
        }

        public static Func<INeuronsLayer, double[], double, double[,]> GetFunc(ActivationFunctionType functionType)
        {
            var derivative = GetDerivative(functionType);

            return (layer, input, rate) =>
            {
                var result = layer.Weights.Clone() as double[,];

                for(int i= 0; i < layer.NeuronsCount; i++)
                {
                    for(int j = 0; j < input.Length; j++)
                    {
                        result[j, i] += rate * layer.Error[i] * input[j] * derivative(layer.Value[i]);
                    }
                }

                return result;
            };
        }

        private static Func<double, double> GetFunction(ActivationFunctionType type)
        {
            return type switch
            {
                ActivationFunctionType.Linear => (i => i),
                ActivationFunctionType.Sigmoid => (i => 1 / (1 + Math.Exp(-i))),
                ActivationFunctionType.TanH => (i => Math.Tanh(i)),
                ActivationFunctionType.ReLU => (i => Math.Max(0, i)),
                ActivationFunctionType.LReLU => (i => Math.Max(i * 0.01, i))
            };
        }

        private static Func<double, double> GetDerivative(ActivationFunctionType functionType)
        {
            return functionType switch
            {
                ActivationFunctionType.Linear => (i => 1),
                ActivationFunctionType.Sigmoid => (i => i * (1 - i)),
                ActivationFunctionType.TanH => (i => 1 - Math.Pow(i, 2)),
                ActivationFunctionType.ReLU => (i => i > 0 ? 1 : 0),
                ActivationFunctionType.LReLU => (i => i > 0 ? 1 : 0.01)
            };
        }
    }
}