using Microsoft.VisualStudio.TestTools.UnitTesting;

using NNet.Functions;

namespace NNet.Test.Network
{
    [TestClass]
    public class NetworkTests
    {
        [TestMethod]
        public void IsConvertCorrect_InputIsBuilderResult_ResultIsConvertedNetwork()
        {
            var test = NeuralNetwork.Builder.SetInputSize(2)
                                            .AddLayer(1, ActivationFunctionType.Linear)
                                            .AddLayer(2, ActivationFunctionType.LReLU)
                                            .Build();

            NeuralNetwork result = test;

            Assert.AreEqual(2, result.LayersCount);
            Assert.AreEqual(2, result.InputSize);
            Assert.AreEqual(2, result.OutputSize);

            Assert.IsTrue(result.Layers[0].InputSize == 2);
            Assert.IsTrue(result.Layers[0].NeuronsCount == 1);
            Assert.IsTrue(result.Layers[0].ActivationFunction == ActivationFunctionType.Linear);

            Assert.IsTrue(result.Layers[1].InputSize == 1);
            Assert.IsTrue(result.Layers[1].NeuronsCount == 2);
            Assert.IsTrue(result.Layers[1].ActivationFunction == ActivationFunctionType.LReLU);
        }

        [TestMethod]
        public void IsCalculateCorrect_InputIs1And1_ResultIs2()
        {
            NeuralNetwork test = NeuralNetwork.Builder.SetInputSize(2)
                                                      .AddLayer(1, ActivationFunctionType.Linear)
                                                      .Build();

            double[,] TEST_WEIGHTS = { { 1 }, { 1 } };
            double[] TEST_INPUT = { 1, 1 };
            double[] TEST_BIAS = { 0 };
            double CORRECT_OUTPUT = 2;

            test.Layers[0].Weights = TEST_WEIGHTS;
            test.Layers[0].Bias = TEST_BIAS;

            var result = test.GetResult(TEST_INPUT);

            Assert.AreEqual(CORRECT_OUTPUT, result[0]);
        }

        [TestMethod]
        public void IsLearnCorrect_InputErrorIs0And2_ResultIsCorrectWeightsAndBias()
        {
            NeuralNetwork test = NeuralNetwork.Builder.SetInputSize(2)
                                                      .AddLayer(2, ActivationFunctionType.Linear)
                                                      .Build();

            double[,] TEST_WEIGHTS = { { 1, 1 }, { 1, 1 } };
            double[] TEST_INPUT = { 1, -1 };
            double[] TEST_BIAS = { 0, 0 };
            double[] TEST_ERROR = { 0, 2 };
            double TEST_RATE = 0.5;

            double[,] CORRECT_WEIGHTS = { { 1, 2 }, { 1, 0 } };
            double[] CORRECT_BIAS = { 0, 1 };

            test.Layers[0].Weights = TEST_WEIGHTS;
            test.Layers[0].Bias = TEST_BIAS;

            test.GetResult(TEST_INPUT);
            test.Learn(TEST_ERROR, TEST_RATE);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, test.Layers[0].Weights);
            CollectionAssert.AreEqual(CORRECT_BIAS, test.Layers[0].Bias);
        }
    }
}