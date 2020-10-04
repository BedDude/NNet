using System;

using NNet.Functions;

namespace NNet.Neurons
{
    /// <summary>
    /// The class of the layer of the neurons.
    /// </summary>
    public class NeuronsLayer : INeuronsLayer
    {
        /// <summary>
        /// Type of the activation function.
        /// </summary>
        private ActivationFunctionType _functionType;

        /// <summary>
        /// The activation function.
        /// </summary>
        /// <remarks>
        /// The first parameter is the input data.
        /// The second parameter is the layer of the neurons.
        /// The result is the values of the neurons in the layer.
        /// </remarks>
        private Func<double[], INeuronsLayer, double[]> _activationFunction;

        /// <summary>
        /// The training function.
        /// </summary>
        /// <remarks>
        /// The first parameter is the layer of the neurons.
        /// The second parameter is the layer's input data.
        /// The third parameter is the rate of the learning.
        /// The result is the new weights - new biases pair.
        /// </remarks>
        private Func<INeuronsLayer, double[], double, (double[,], double[])> _trainingFunction;


        /// <summary>
        /// The count of the neurons in the layer.
        /// </summary>
        public int NeuronsCount { get; private set; }

        /// <summary>
        /// The size of the input data.
        /// </summary>
        public int InputSize { get; private set; }

        /// <summary>
        /// The values of the neurons in the layer.
        /// </summary>
        public double[] Value { get; private set; }

        /// <summary>
        /// The biases of the neurons in the layer.
        /// </summary>
        public double[] Bias { get; set; }

        /// <summary>
        /// The errors of the naurons in the layer.
        /// </summary>
        public double[] Error { get; set; }

        /// <summary>
        /// The weights of the layer.
        /// </summary>
        public double[,] Weights { get; set; }

        /// <summary>
        /// Type of the activation function.
        /// </summary>
        public ActivationFunctionType ActivationFunction
        {
            get => _functionType;
            set
            {
                _functionType = value;

                _activationFunction = CalculationFunctions.GetActivationFunction(value);
                _trainingFunction = CalculationFunctions.GetTrainingFunction(value);
            }
        }


        /// <summary>
        /// Initializes a new instance of <see cref="NeuronsLayer"/>.
        /// </summary>
        /// <param name="size">The count of the neurons in the layer.</param>
        /// <param name="inputSize">The size of the input data.</param>
        public NeuronsLayer(int size, int inputSize)
        {
            NeuronsCount = size;
            InputSize = inputSize;

            Value = new double[size];
            Bias = new double[size];
            Error = new double[size];
            Weights = new double[inputSize, size];

            var rnd = new Random();
            for(int i = 0; i < size; i++)
            {
                Bias[i] = rnd.NextDouble() - 0.5;
                for(int j = 0; j < inputSize; j++)
                {
                    Weights[j, i] = rnd.NextDouble() - 0.5;
                }
            }
        }

        /// <summary>
        /// Calculates the new values of the neurons in the layer.
        /// </summary>
        /// <param name="input">The input data.</param>
        public void Active(double[] input)
        {
            Value = _activationFunction?.Invoke(input, this);
        }

        /// <summary>
        /// Changes the weights of the layer.
        /// </summary>
        /// <remarks>
        /// It is uses the backpropogation method.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="rate">The rate of the learning.</param>
        public void Learn(double[] input, double rate)
        {
            var result = _trainingFunction?.Invoke(this, input, rate);

            Weights = result.Value.Item1;
            Bias = result.Value.Item2;
        }

        /// <summary>
        /// Translates error to the previous layer.
        /// </summary>
        /// <param name="previousLayer">The previous layer.</param>
        public void TranslateError(INeuronsLayer previousLayer)
        {
            var local = new double[previousLayer.NeuronsCount];
            for (int i = 0; i < previousLayer.NeuronsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    local[i] += Error[j] * Weights[i, j];
                }
            }

            previousLayer.Error = local;
        }
    }
}