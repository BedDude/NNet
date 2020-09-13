using NNet.Functions;

namespace NNet.Neurons
{
    public interface INeuronsLayer
    {
        int NeuronsCount { get; }
        double[] Value { get; }
        double[] Bias { get; set; }
        double[] Error { get; set; }
        double[,] Weights { get; set; }
        ActivationFunctionType ActivationFunction { get; set; }

        void Active(double[] input);
        void Learn(double[] input, double rate);
        void TranslateError();
    }
}