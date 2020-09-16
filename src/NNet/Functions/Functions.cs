using System;

using NNet.Neurons;

namespace NNet.Functions
{
    //пока internal, а там как пойдет
    internal static class CalculationFunctions
    {
        public static Func<double[], double[,], double[]> GetActivationFunction(ActivationFunctionType functionType)
        {
            throw new NotImplementedException();
        }

        public static Func<NeuronsLayer, double[], double, double[,]> GetFunc(ActivationFunctionType functionType)
        {
            throw new NotImplementedException();
        }
    }
}