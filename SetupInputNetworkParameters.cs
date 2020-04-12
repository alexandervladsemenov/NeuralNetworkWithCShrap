using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CreatingTestCodeLibrary
{
    public static class SetupInputNetworkParameters
    {

        public static void SetUpNNParameters(out NetWorkMethodsAndParameters<Matrix<float>, Vector<float>> parameters)
        {
            parameters = new NetWorkMethodsAndParameters<Matrix<float>, Vector<float>>();// creaty empty;


        }
        public static void SetUpNNParameters(out NetWorkMethodsAndParameters<float [][] , float []> parameters)
        {
            parameters = null;

        }
    }
}
