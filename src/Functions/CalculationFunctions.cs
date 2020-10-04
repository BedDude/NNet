using System;

using NNet.Neurons;

namespace NNet.Functions
{
    /// <summary>
    /// A functions generator class.
    /// </summary>
    /// <remarks>
    /// It used to activation and training functions generation.
    /// </remarks>
    internal static class CalculationFunctions
    {
        /// <summary>
        /// Generates a function that can calculate the new values of the layer.
        /// </summary>
        /// <param name="functionType">Type of activation function.</param>
        /// <returns>Activation function.</returns>
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
                        sumOfInputs += input[j] * layer.Weights[j, i];
                    }
                    sumOfInputs += layer.Bias[i];

                    result[i] = function(sumOfInputs);
                }

                return result;
            };
        }

        /// <summary>
        /// Generates a function that can change layer's weights.
        /// </summary>
        /// <param name="functionType">Type of activation function.</param>
        /// <returns>The training function.</returns>
        public static Func<INeuronsLayer, double[], double, (double[,], double[])> GetTrainingFunction(ActivationFunctionType functionType)
        {
            var derivative = GetDerivative(functionType);

            return (layer, input, rate) =>
            {
                var result = (layer.Weights.Clone() as double[,], layer.Bias.Clone() as double[]);

                for(int i= 0; i < layer.NeuronsCount; i++)
                {
                    for(int j = 0; j < input.Length; j++)
                    {
                        result.Item1[j, i] += rate * layer.Error[i] * input[j] * derivative(layer.Value[i]);
                    }
                    result.Item2[i] += rate * layer.Error[i] * derivative(layer.Value[i]);
                }

                return result;
            };
        }

        /// <summary>
        /// A function that returns activation function.
        /// </summary>
        /// <param name="type">Type of activation function.</param>
        /// <returns>Selected function.</returns>
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

        /// <summary>
        /// A function that returns derivative of activation function.
        /// </summary>
        /// <param name="functionType">Type of activation function.</param>
        /// <returns>Derivative of selected function.</returns>
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