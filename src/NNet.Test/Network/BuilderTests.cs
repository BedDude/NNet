using Microsoft.VisualStudio.TestTools.UnitTesting;

using NNet;
using NNet.IO;
using NNet.Builder;
using NNet.Neurons;
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
        public void AddLayer_InputIsSizesAndFunctions_ResultsLayersCountIs3()
        {
            var result = NeuralNetwork.Builder.AddNeurons(1, 2, 3)
                                              .AddFunctions(ActivationFunctionType.Linear, 
                                                            ActivationFunctionType.Linear,
                                                            ActivationFunctionType.Linear)
                                              .Build()
                                              .ToNetwork();

            Assert.AreEqual(3, result.LayersCount);
        }

        [TestMethod]
        public void AddLayer_InputIsSizesAndFunctions_ResultsLayersCountIs2()
        {
            var result = NeuralNetwork.Builder.AddNeurons(1, 2, 3, 4, 5)
                                              .AddFunctions(ActivationFunctionType.Linear, 
                                                            ActivationFunctionType.Linear)
                                              .Build()
                                              .ToNetwork();

            Assert.AreEqual(2, result.LayersCount);
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
            var result = NeuralNetwork.Builder.BuildFromConfig(@"./../../../Network/test.ncfg")
                                              .ToNetwork();

            Assert.AreEqual(2, result.InputSize);
            Assert.AreEqual(1, result.LayersCount);

            double[] TEST_BIAS = { 1 };
            double[,] TEST_WEIGHTS = { { 1 }, { 1 } };

            Assert.AreEqual(2, result.Layers[0].InputSize);
            Assert.AreEqual(1, result.Layers[0].NeuronsCount);
            CollectionAssert.AreEqual(TEST_BIAS, result.Layers[0].Bias);
            CollectionAssert.AreEqual(TEST_WEIGHTS, result.Layers[0].Weights);
            Assert.AreEqual(ActivationFunctionType.Linear, result.Layers[0].ActivationFunction);
        }
    }
}