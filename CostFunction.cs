using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    public static class CostFunction
    {
        // matrix Math NET
        public static void GradientQuadratic(out object vaL, Matrix<float> outputExpected, Matrix<float> outputNN, bool grads)
        {

            if (grads)
                vaL = outputNN.Subtract(outputExpected); // gradient
            else
                vaL = Math.Pow(outputExpected.Subtract(outputNN).FrobeniusNorm(),2); // total loss/cost function
        }
        public static void GradientEntropy(out object vaL, Matrix<float> outputExpected, Matrix<float> outputNN, bool grads)
        {
            Func<float, float> LogMinusOneMinusLog = (float a) => (float)(Math.Log(a) - Math.Log(1.0 - a));
            Func<float, float> OneMinusLog = (float a) => (float)(Math.Log(1 - a));
            Func<float, float> OneMinus = (float a) => (1 - a);
            if (grads) // we need cross entropy now
                vaL = outputExpected.PointwiseDivide(outputNN) - (outputExpected.Map(OneMinus)).PointwiseDivide(outputNN.Map(OneMinus));
            else
                vaL = (outputExpected.PointwiseMultiply(outputNN.Map(LogMinusOneMinusLog)).Add(outputNN.Map(OneMinusLog))).L1Norm(); // need to work this out
        }
        public static void GradientModule(out object vaL, Matrix<float> outputExpected, Matrix<float> outputNN, bool grads)
        {
            vaL = null;
        }
        public static void GradientSoftMax(out object vaL, Matrix<float> outputExpected, Matrix<float> outputNN, bool grads)
        {
            vaL = null;
        }
        // for cuda float [][] and float [,]
    }
}
