using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NNet.Neurons;
using NNet.Functions;

namespace NNet.Test.Neurons
{
    [TestClass]
    public class NeuronsTest
    {
        private INeuronsLayer _layer;

        public NeuronsTest()
        {
            _layer = new NeuronsLayer(1, 1);

            double[,] TEST_WEIGHTS = { { 1 } };
            double[] TEST_BIAS = { 0 };
            double[] TEST_VALUE = { 0.5 };

            _layer.Weights = TEST_WEIGHTS;
            _layer.Bias = TEST_BIAS;

            _layer.ActivationFunction = ActivationFunctionType.Linear;
            _layer.Active(TEST_VALUE);
        }

        [TestMethod]
        public void ValueIsCalculatedCorect_Linear()
        {
            double[] TEST_INPUT_1 = { 1 };
            double[] TEST_INPUT_2 = { 0 };
            double[] TEST_INPUT_3 = { -1 };

            double[] TEST_OUTPUT_1 = { 1 };
            double[] TEST_OUTPUT_2 = { 0 };
            double[] TEST_OUTPUT_3 = { -1 };

            _layer.ActivationFunction = ActivationFunctionType.Linear;

            _layer.Active(TEST_INPUT_1);
            CollectionAssert.AreEqual(TEST_OUTPUT_1, _layer.Value);

            _layer.Active(TEST_INPUT_2);
            CollectionAssert.AreEqual(TEST_OUTPUT_2, _layer.Value);

            _layer.Active(TEST_INPUT_3);
            CollectionAssert.AreEqual(TEST_OUTPUT_3, _layer.Value);
        }

        [TestMethod]
        public void ValueIsCalculatedCorect_Sigmoid()
        {
            double[] TEST_INPUT_1 = { 1 };
            double[] TEST_INPUT_2 = { 0 };
            double[] TEST_INPUT_3 = { -1 };

            double[] TEST_OUTPUT_1 = { 0.73 };
            double[] TEST_OUTPUT_2 = { 0.5 };
            double[] TEST_OUTPUT_3 = { 0.27 };

            _layer.ActivationFunction = ActivationFunctionType.Sigmoid;

            _layer.Active(TEST_INPUT_1);
            var res1 = _layer.Value;
            res1[0] = Math.Round(res1[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_1, res1);

            _layer.Active(TEST_INPUT_2);
            var res2 = _layer.Value;
            res2[0] = Math.Round(res2[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_2, res2);

            _layer.Active(TEST_INPUT_3);
            var res3 = _layer.Value;
            res3[0] = Math.Round(res3[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_3, res3);
        }

        [TestMethod]
        public void ValueIsCalculatedCorect_TanH()
        {
            double[] TEST_INPUT_1 = { 1 };
            double[] TEST_INPUT_2 = { 0 };
            double[] TEST_INPUT_3 = { -1 };

            double[] TEST_OUTPUT_1 = { 0.76 };
            double[] TEST_OUTPUT_2 = { 0 };
            double[] TEST_OUTPUT_3 = { -0.76 };

            _layer.ActivationFunction = ActivationFunctionType.TanH;

            _layer.Active(TEST_INPUT_1);
            var res1 = _layer.Value;
            res1[0] = Math.Round(res1[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_1, res1);

            _layer.Active(TEST_INPUT_2);
            var res2 = _layer.Value;
            res2[0] = Math.Round(res2[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_2, res2);

            _layer.Active(TEST_INPUT_3);
            var res3 = _layer.Value;
            res3[0] = Math.Round(res3[0], 2);
            CollectionAssert.AreEqual(TEST_OUTPUT_3, res3);
        }

        [TestMethod]
        public void ValueIsCalculatedCorect_ReLU()
        {
            double[] TEST_INPUT_1 = { 1 };
            double[] TEST_INPUT_2 = { 0 };
            double[] TEST_INPUT_3 = { -1 };

            double[] TEST_OUTPUT_1 = { 1 };
            double[] TEST_OUTPUT_2 = { 0 };
            double[] TEST_OUTPUT_3 = { 0 };

            _layer.ActivationFunction = ActivationFunctionType.ReLU;

            _layer.Active(TEST_INPUT_1);
            CollectionAssert.AreEqual(TEST_OUTPUT_1, _layer.Value);

            _layer.Active(TEST_INPUT_2);
            CollectionAssert.AreEqual(TEST_OUTPUT_2, _layer.Value);

            _layer.Active(TEST_INPUT_3);
            CollectionAssert.AreEqual(TEST_OUTPUT_3, _layer.Value);
        }

        [TestMethod]
        public void ValueIsCalculatedCorect_LReLU()
        {
            double[] TEST_INPUT_1 = { 1 };
            double[] TEST_INPUT_2 = { 0 };
            double[] TEST_INPUT_3 = { -1 };

            double[] TEST_OUTPUT_1 = { 1 };
            double[] TEST_OUTPUT_2 = { 0 };
            double[] TEST_OUTPUT_3 = { -0.01 };

            _layer.ActivationFunction = ActivationFunctionType.LReLU;

            _layer.Active(TEST_INPUT_1);
            CollectionAssert.AreEqual(TEST_OUTPUT_1, _layer.Value);

            _layer.Active(TEST_INPUT_2);
            CollectionAssert.AreEqual(TEST_OUTPUT_2, _layer.Value);

            _layer.Active(TEST_INPUT_3);
            CollectionAssert.AreEqual(TEST_OUTPUT_3, _layer.Value);
        }

        [TestMethod]
        public void ErrorTranclatedCorrect()
        {
            var uselessLayer = new NeuronsLayer(2, 1);

            double[,] TEST_WEIGHTS = { { 1, 0.5 } };
            double[] TEST_ERROR = { 1, -0.5 };
            double[] CORRECT_ERROR = { 0.75 };

            uselessLayer.Weights = TEST_WEIGHTS;
            uselessLayer.Error = TEST_ERROR;

            uselessLayer.TranslateError(_layer);

            CollectionAssert.AreEqual(CORRECT_ERROR, _layer.Error);
        }

        [TestMethod]
        public void WeightsIsUpdatedCorrect_Linear()
        {
            double[] TEST_ERROR = { 0.5 };
            double[] TEST_VALUE = { 0.5 };
            double[] CORRECT_WEIGHTS = { 1.25 };

            _layer.ActivationFunction = ActivationFunctionType.Linear;
            _layer.Error = TEST_ERROR;

            _layer.Learn(TEST_VALUE, 1);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, _layer.Weights);
        }

        [TestMethod]
        public void WeightsIsUpdatedCorrect_Sigmoid()
        {
            double[] TEST_ERROR = { 0.5 };
            double[] TEST_VALUE = { 0.5 };
            double[] CORRECT_WEIGHTS = { 1.0625 };

            _layer.ActivationFunction = ActivationFunctionType.Sigmoid;
            _layer.Error = TEST_ERROR;

            _layer.Learn(TEST_VALUE, 1);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, _layer.Weights);
        }

        [TestMethod]
        public void WeightsIsUpdatedCorrect_TanH()
        {
            double[] TEST_ERROR = { 0.5 };
            double[] TEST_VALUE = { 0.5 };
            double[] CORRECT_WEIGHTS = { 1.1875 };

            _layer.ActivationFunction = ActivationFunctionType.TanH;
            _layer.Error = TEST_ERROR;

            _layer.Learn(TEST_VALUE, 1);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, _layer.Weights);
        }

        [TestMethod]
        public void WeightsIsUpdatedCorrect_ReLU()
        {
            double[] TEST_ERROR = { 0.5 };
            double[] TEST_VALUE = { 0.5 };
            double[] CORRECT_WEIGHTS = { 1.25 };

            _layer.ActivationFunction = ActivationFunctionType.Linear;
            _layer.Error = TEST_ERROR;

            _layer.Learn(TEST_VALUE, 1);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, _layer.Weights);
        }

        [TestMethod]
        public void WeightsIsUpdatedCorrect_LReLU()
        {
            double[] TEST_ERROR = { 0.5 };
            double[] TEST_VALUE = { 0.5 };
            double[] CORRECT_WEIGHTS = { 1.25 };

            _layer.ActivationFunction = ActivationFunctionType.Linear;
            _layer.Error = TEST_ERROR;

            _layer.Learn(TEST_VALUE, 1);

            CollectionAssert.AreEqual(CORRECT_WEIGHTS, _layer.Weights);
        }
    }
}