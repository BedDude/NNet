using System;

using NNet.Functions;

namespace NNet.Neurons
{
    public class NeuronsLayer : INeuronsLayer
    {
        private ActivationFunctionType _functionType;
        private Func<double[], INeuronsLayer, double[]> _activationFunction;
        private Func<INeuronsLayer, double[], double, (double[,], double[])> _trainingFunction;

        public int NeuronsCount { get; private set; }
        public int InputSize { get; private set; }
        public double[] Value { get; private set; }
        public double[] Bias { get; set; }
        public double[] Error { get; set; }
        public double[,] Weights { get; set; }
        public ActivationFunctionType ActivationFunction
        {
            get => _functionType;
            set
            {
                _functionType = value;

                _activationFunction = CalculationFunctions.GetActivationFunction(value);
                _trainingFunction = CalculationFunctions.GetFunc(value);
            }
        }

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

        public void Active(double[] input)
        {
            Value = _activationFunction?.Invoke(input, this);
        }

        public void Learn(double[] input, double rate)
        {
            var result = _trainingFunction?.Invoke(this, input, rate);

            Weights = result.Value.Item1;
            Bias = result.Value.Item2;
        }

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