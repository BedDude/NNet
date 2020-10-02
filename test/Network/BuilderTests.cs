using Microsoft.VisualStudio.TestTools.UnitTesting;

using NNet.IO;
using NNet.Functions;

namespace NNet.Test.Network
{
    [TestClass]
    public class BuilderTests
    {
        [TestMethod]
        public void SetInputSize_InputIs5_ResultsIntputSizeIs5()
        {
            var result = NeuralNetwork.Builder.SetInputSize(5)
                                              .Build()
                                              .ToNetwork();

            Assert.AreEqual(5, result.InputSize);
        }

        [TestMethod]
        public void AddLayer_InputIsLayersInformation_ResultsLayersCountIs3()
        {
            var result = NeuralNetwork.Builder.AddLayer(1, ActivationFunctionType.Linear)
                                              .AddLayer(2, ActivationFunctionType.Linear)
                                              .AddLayer(3, ActivationFunctionType.Linear)
                                              .Build()
                                              .ToNetwork();

            Assert.AreEqual(3, result.LayersCount);
        }

        [TestMethod]
        public void SetSerializer_InputIsNone_ReturnIsNull()
        {
            var result = NeuralNetwork.Builder.SetSerializer(SerializerType.None)
                                              .Build()
                                              .ToNetwork();

            Assert.IsFalse(result.Serializer is ConfigSerializer);
        }

        [TestMethod]
        public void SetSerializer_InputIsConfigSerializer_ReturnIsConfigSerializer()
        {
            var result = NeuralNetwork.Builder.SetSerializer(SerializerType.ConfigSerializer)
                                              .Build()
                                              .ToNetwork();

            Assert.IsTrue(result.Serializer is ConfigSerializer);
        }

        [TestMethod]
        public void IsBuildCorrect()
        {
            var result = NeuralNetwork.Builder.AddLayer(3, ActivationFunctionType.Sigmoid)
                                              .SetInputSize(2)
                                              .AddLayer(1, ActivationFunctionType.Sigmoid)
                                              .Build()
                                              .ToNetwork();

            Assert.IsTrue(result.LayersCount == 2);
            Assert.IsTrue(result.InputSize == 2);
            Assert.IsTrue(result.OutputSize == 1);

            Assert.IsTrue(result.Layers[0].NeuronsCount == 3);
            Assert.IsTrue(result.Layers[0].ActivationFunction == ActivationFunctionType.Sigmoid);

            Assert.IsTrue(result.Layers[1].NeuronsCount == 1);
            Assert.IsTrue(result.Layers[1].ActivationFunction == ActivationFunctionType.Sigmoid);
        }

        [TestMethod]
        public void BuildFromConfigFile()
        {
            double[] TEST_BIAS = { 1 };
            double[,] TEST_WEIGHTS = { { 1 }, { 1 } };
            NeuralNetwork TEST_NETWORK = NeuralNetwork.Builder.SetInputSize(2)
                                                    .SetSerializer(SerializerType.ConfigSerializer)
                                                    .AddLayer(1, ActivationFunctionType.Linear)
                                                    .Build();

            TEST_NETWORK.Layers[0].Weights = TEST_WEIGHTS;
            TEST_NETWORK.Layers[0].Bias = TEST_BIAS;

            TEST_NETWORK.SaveNetwork(@"./../../../Network", "test");
            
            var result = NeuralNetwork.Builder.BuildFromConfigFile(@"./../../../Network/test.ncfg")
                                              .ToNetwork();

            Assert.AreEqual(2, result.InputSize);
            Assert.AreEqual(1, result.LayersCount);

            Assert.AreEqual(2, result.Layers[0].InputSize);
            Assert.AreEqual(1, result.Layers[0].NeuronsCount);
            CollectionAssert.AreEqual(TEST_BIAS, result.Layers[0].Bias);
            CollectionAssert.AreEqual(TEST_WEIGHTS, result.Layers[0].Weights);
            Assert.AreEqual(ActivationFunctionType.Linear, result.Layers[0].ActivationFunction);
        }

        [TestMethod]
        public void BuildFromNetworkFile()
        {
            double[] TEST_BIAS = { 1 };
            double[,] TEST_WEIGHTS = { { 1 }, { 1 } };
            NeuralNetwork TEST_NETWORK = NeuralNetwork.Builder.SetInputSize(2)
                                                    .SetSerializer(SerializerType.NetworkSerializer)
                                                    .AddLayer(1, ActivationFunctionType.Linear)
                                                    .Build();

            TEST_NETWORK.Layers[0].Weights = TEST_WEIGHTS;
            TEST_NETWORK.Layers[0].Bias = TEST_BIAS;

            TEST_NETWORK.SaveNetwork(@"./../../../Network", "test");

            var result = NeuralNetwork.Builder.BuildFromNetworkFile(@"./../../../Network/test.unf")
                                              .ToNetwork();

            Assert.AreEqual(2, result.InputSize);
            Assert.AreEqual(1, result.LayersCount);

            Assert.AreEqual(2, result.Layers[0].InputSize);
            Assert.AreEqual(1, result.Layers[0].NeuronsCount);
            CollectionAssert.AreEqual(TEST_BIAS, result.Layers[0].Bias);
            CollectionAssert.AreEqual(TEST_WEIGHTS, result.Layers[0].Weights);
            Assert.AreEqual(ActivationFunctionType.Linear, result.Layers[0].ActivationFunction);
        }
    }
}