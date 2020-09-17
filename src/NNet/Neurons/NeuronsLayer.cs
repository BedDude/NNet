using System;

using NNet.Functions;

namespace NNet.Neurons
{
    public class NeuronsLayer : INeuronsLayer
    {
        private ActivationFunctionType _functionType;
        private Func<double[], INeuronsLayer, double[]> _activationFunction;
        private Func<INeuronsLayer, double[], double, double[,]> _trainingFunction;

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
        }

        public void Active(double[] input)
        {
            Value = _activationFunction?.Invoke(input, this);
        }

        public void Learn(double[] input, double rate)
        {
            Weights = _trainingFunction?.Invoke(this, input, rate);
        }

        public void TranslateError(INeuronsLayer previousLayer)
        {
            var local = new double[NeuronsCount];
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