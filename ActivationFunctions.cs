using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
namespace CreatingTestCodeLibrary
{// here we store activation function applied to matrices/vectors. We do need a separate class.
    public delegate void ReturnActiveFunctions<T>( out ActivationFunction<T>[] activationFunctions, int layerIndex, params string[] inputNames); // need to test
    public static class ActivationFunctions
    {
        private static FunctionActivate ActivationFunctionHidden; // FunctionActivate == Func<float,float>
        private static FunctionActivate ActivationFunctionOutput;
        private static FunctionActivate ActivationFunctionHiddenDer;
        private static FunctionActivate ActivationFunctionOutputDer;
        private static ActivationFunction<Matrix<float>>[] ActivationFunction = new ActivationFunction<Matrix<float>>[2];


        public static void SetActivationFunctions(string inputNameHidden, string inputNameOutput, params string[] otherLayers)
        {
            ActivationFunctionHidden = ActiveFunctions.ReturnMethod(inputNameHidden)[0]; // activation function for the hidden layer
            ActivationFunctionHiddenDer = ActiveFunctions.ReturnMethod(inputNameHidden)[1]; // activation function derivative for the hidden layer
            ActivationFunctionOutput = ActiveFunctions.ReturnMethod(inputNameOutput)[0];// same but for output layer
            ActivationFunctionOutputDer = ActiveFunctions.ReturnMethod(inputNameOutput)[1];// same but for output layer, der
        }
        public static void ReturnActivationFunction(out ActivationFunction<Matrix<float>>[] ActivationFunction, int layerIndex, params string[] inputNames  )
        {
            ActivationFunction = new ActivationFunction<Matrix<float>>[2]; // activationfunction allocation
            SetActivationFunctions(inputNames[0], inputNames[1]);
            Func<float, float>[] f = new Func<float, float> [2];
            if (layerIndex == 0)
            {
                f[0] = (float x) => ActivationFunctionHidden(x); // passing activation functions
                f[1] = (float x) => ActivationFunctionHiddenDer(x); //passing activation functions derivative // hidden layers
            }
            else
            {
                f[0] = (float x) => ActivationFunctionOutput(x); // passing activation functions
                f[1] = (float x) => ActivationFunctionOutputDer(x);////passing activation functions derivative // output layers
            }
            ActivationFunction[0] = (Matrix<float> m) => m.Map(f[0]);
            ActivationFunction[1] = (Matrix<float> m) => m.Map(f[1]);

        }
        public static void ReturnActivationFunction(out ActivationFunction<float[,]>[] ActivationFunction, int layerIndex, params string[] inputNames) // this is for CUDA sample
        {
            ActivationFunction = new ActivationFunction<float[,]>[2];
        }
        public static void ReturnActivationFunction(out ActivationFunction<float[][]>[] ActivationFunction, int layerIndex, params string[] inputNames) // this is for CUDA sample
        {
            ActivationFunction = new ActivationFunction<float[][]>[2];
        }

    }
}
