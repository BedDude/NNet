using NNet.Functions;

namespace NNet.Neurons
{
    /// <summary>
    /// Provides means to interacting with the layer of the neurons.
    /// </summary>
    public interface INeuronsLayer
    {
        /// <summary>
        /// The count of the neurons in the layer.
        /// </summary>
        int NeuronsCount { get; }

        /// <summary>
        /// The size of the input data.
        /// </summary>
        int InputSize { get; }

        /// <summary>
        /// The values of the neurons in the layer.
        /// </summary>
        double[] Value { get; }

        /// <summary>
        /// The biases of the neurons in the layer.
        /// </summary>
        double[] Bias { get; set; }

        /// <summary>
        /// The errors of the naurons in the layer.
        /// </summary>
        double[] Error { get; set; }

        /// <summary>
        /// The weights of the layer.
        /// </summary>
        double[,] Weights { get; set; }

        /// <summary>
        /// Type of the activation function.
        /// </summary>
        ActivationFunctionType ActivationFunction { get; set; }


        /// <summary>
        /// Calculates the new values of the neurons in the layer.
        /// </summary>
        /// <param name="input"></param>
        void Active(double[] input);

        /// <summary>
        /// Changes the weights of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="rate"></param>
        void Learn(double[] input, double rate);

        /// <summary>
        /// Translates error to the previous layer.
        /// </summary>
        /// <param name="previousLayer">The previous layer.</param>
        void TranslateError(INeuronsLayer previousLayer);
    }
}