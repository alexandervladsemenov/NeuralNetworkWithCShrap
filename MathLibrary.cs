using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using System.Collections.Generic;
using System.Linq;
using DenseMatrix = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix;
using DenseVector = MathNet.Numerics.LinearAlgebra.Single.DenseVector;

namespace CreatingTestCodeLibrary
{
    public delegate float FunctionActivate(float input);
    public static class MathLibrary
    {
        // Matrix Multiplications
        public static Matrix<float> GenerateMatrix(int rows, int colums)
        {
            return DenseMatrix.Create(rows, colums, 0);
        }
        public static Matrix<float> GenerateMatrix(Matrix<float> Sample)
        {
            int rows, columns;
            DimsOfMatrix(Sample.ToTypeString(), out rows, out columns);
            int[] maxIndex = new int[columns];
            float[] maxValue= new float[columns];
            Func<int, int, float> Filter = (int j, int i) => j == maxIndex[i] ? 1 : 0;
            for (int i = 0; i < columns; i++)
                for (int j = 0; j < rows; j++)
                {
                    maxIndex[i] = maxValue[i] < Sample[j, i] ? j : maxIndex[i];
                    maxValue[i] = maxValue[i] < Sample[j, i] ? Sample[j, i] : maxValue[i];
                } // work on it further // the plan is simle// classical trajectoreis, supervised learning
            // here we need to figure out how to extract dimenstions from a matrix
            return DenseMatrix.Create(rows, columns, Filter);
        }
        public static object ValidatePrediction(Matrix<float> ExpectedOutuput, Matrix<float> ComputedOutput)
        {
            int numberOfSuccefullRuns = 0;
            int j,rowsOne, columnsOne;
            int rowsTwo, columnsTwo;
            DimsOfMatrix(ExpectedOutuput.ToTypeString(), out rowsOne, out columnsOne);
            DimsOfMatrix(ComputedOutput.ToTypeString(), out rowsTwo, out columnsTwo);
            if (rowsOne != rowsTwo && columnsOne != columnsTwo)
                throw new Exception("dimensions do not match");
            for (int i = 0; i < columnsOne; i++)
            {
                for (j = 0; j < rowsOne; j++)
                {
                    if (ExpectedOutuput[j, i] != ComputedOutput[j, i])
                        break;
                } // work on it further // the plan is simle// classical trajectoreis, supervised learning
                if (j == rowsOne)
                    numberOfSuccefullRuns++;
            }
            return numberOfSuccefullRuns;
        }
        public static void DimsOfMatrix(string input, out int rows, out int columns)
        {
            string rowsStr = string.Empty;
            string columnsStr = string.Empty;
            int positionOfX = 0;
            for (int i = 0; i < input.Length; i++)
            {
                if (Char.IsDigit(input[i])) // recording number of rows
                {
                    rowsStr += input[i];
                }
                positionOfX++;
                if (input[i] == 'x' && Char.IsDigit(input[i - 1]))
                {
                    break;
                }
            }
            for (int i = positionOfX; i < input.Length; i++)// recording number of coulumns
            {
                if (Char.IsDigit(input[i]))
                    columnsStr += input[i];
            }

            rows = Int32.Parse(rowsStr);
            columns = Int32.Parse(columnsStr);
        }
        public static Matrix<double> MatMatMul( Matrix<double> M1, Matrix<double> M2)
        {
            return M1.Multiply(M2);// M1[n1,n2] x M2[n2,n3] = M3[n1,n3]
        }
        public static Matrix<float> MatMatMul(Matrix<float> M1, Matrix<float> M2)
        {
            return M1.Multiply(M2);// M1[n1,n2] x M2[n2,n3] = M3[n1,n3]
        }
        public static Matrix<float> MatMatMulPointWise(Matrix<float> M1, Matrix<float> M2)
        {
            return M1.PointwiseMultiply(M2);// M1[n1,n2] x M2[n1,n2] = M3[n1,n2]
        }
        public static Matrix<float> MatMatMulTranspose(Matrix<float> M1, Matrix<float> M2)
        {
            return M1.TransposeThisAndMultiply(M2);// M1[n2,n1]T x M2[n2,n3] = M3[n1,n3]
        }
        public static Matrix<float> MatMatMulSecondTranspose(Matrix<float> M1, Matrix<float> M2)
        {
            return M1.TransposeAndMultiply(M2);// M1[n1,n2] x M2[n3,n2]T = M3[n1,n3]
        }
        public static Matrix<float> MatScalarMul(Matrix <float> inputM, object scalar)
        {
            float mult = (float)((double)scalar);
            return inputM.Multiply(mult); // scalar x inputM[n1, n2];
        }
        public static Vector<float> VecScalarMul(Vector<float> inputV, object scalar)
        {
            float mult = (float)((double)scalar);
            return inputV.Multiply(mult);// scalar x inputV[n];
        }
        public static Matrix<int> MatMatMul(Matrix<int> M1, Matrix<int> M2)
        {
            return M1.Multiply(M2);// M1[n1,n2] x M2[n2,n3] = M3[n1,n3]
        }
        // Matrix Summations
        public static Matrix<double> MatMatSum(Matrix<double> M1, Matrix<double> M2)
        {
            return M1.Add(M2);// M1[n1,n2] + M2[n1,n2] = M3[n1,n2]
        }
        public static Matrix<float> MatMatSum(Matrix<float> M1, Matrix<float> M2)
        {
            return M1.Add(M2);// M1[n1,n2] + M2[n1,n2] = M3[n1,n2]
        }
        public static Matrix<int> MatMatSum(Matrix<int> M1, Matrix<int> M2)
        {
            return M1.Add(M2);// M1[n1,n2] + M2[n1,n2] = M3[n1,n2];
        }
        public static Vector<float> VectorVectorSum(Vector<float> V1, Vector<float> V2)
        {
            return V1.Add(V2);// V1[n] + V2[n] = V3[n];
        }
        // copying vector basis in several samples
        public static Matrix<float> VectorToMatrix(int columns, Vector<float> v)
        {
            List<Vector<float>> vv = new List<Vector<float>>(columns); // creating a list  of  column vectors
            for (int i = 0; i < columns; i++)
                vv.Add(v); // initilizaing the matrix List
            return DenseMatrix.OfColumnVectors(vv); // return the column list
        }

        public static Vector<float> GenerateUnitVector(int rows) // generate unit vector
        {
            return DenseVector.Create(rows, 1);
        }
        public static Vector<float> MatVecMul(Matrix<float> m, Vector<float> v) // generate unit vector
        {
            return m.Multiply(v); // M [n1, n2] x v[n2] = v[n1]
        }

        public static Func<Matrix<float>, Matrix<float>> TranposeMatrix = (Matrix<float> m) => m.Transpose();
    }
    public static class RandomGenerators
    {
        // 1 - normal distribution
        // 2 - uniform distribution
        public static void CreateDistribution(params int [] parameters) // generate distribtuion
        {
            double mean = meanDistr[0]; // mean
            double stddev = sigmaDistr[0]; // standart deviation
            if(parameters[0]<0)
            continuousDistribution = new ContinuousUniform(-stddev, stddev, wH2006);// distribution, uniform
            else
            continuousDistribution = new Normal(mean, stddev, wH2006); // distribution, normal

        }
        public static WH2006 wH2006 = new WH2006(); //random number generator
        public static float[] sigmaDistr{ set; get; } // sigma random
        public static float[] meanDistr { set; get; } // sigma meanDistr
        private static IContinuousDistribution continuousDistribution; // distribtuion
        // Generate Random Matrix
        public static Matrix<float> GenerateRandomMatrix(int rows, int columns , params int[] parameters)
        {
            if (parameters.Length < 1)
                throw new Exception("Pass no parameters"); // check if at least 1 parameter has been passed
            meanDistr = new float[] { 0 }; // meanValues is zero
            sigmaDistr = new float[] { (float)Math.Sqrt(1.0/ 1) }; // sum of sigmas
            CreateDistribution(parameters); // generate distirbiution
            return DenseMatrix.CreateRandom(rows, columns, continuousDistribution);// create a matrix with normal/unifrom distribution
        }
        // generating random vector
        public static Vector<float> GenerateRandomVector(int rows, params int[] parameters)
        {
            if (parameters.Length < 1)
                throw new Exception("Pass no parameters");// check if at least 1 parameter has been passed
            meanDistr = new float[] { 0 }; // meanValues is zero
            sigmaDistr = new float[] { (float)Math.Sqrt(1.0 / 1) }; // sum of sigmas
            CreateDistribution(parameters); // generate distirbiution
            return DenseVector.CreateRandom(rows, continuousDistribution);// create a matrix from continous distribtion;
        }
    }
    // create a storage of activation functions and their derivatives
    public static class ActiveFunctions
    {
        public static FunctionActivate[] ReturnMethod(string inputName)
        {
            FunctionActivate[] activeFunctions = new FunctionActivate[2];
            switch (inputName)
            {
                case "Tanh": // passing hyperbolic tanges
                    activeFunctions[0] = (float x) => (Tanh(x)+1)/2; // Tanh(x)
                    activeFunctions[1] = (float x) => (1 - Tanh(x) * Tanh(x))/2; // dTanh(x)/dx
                    break;
                case "Sigmoid":
                    activeFunctions[0] = (float x) => (float) (1.0 / (1.0 + Math.Exp(-x)));// 1/(1+exp(-x));
                    activeFunctions[1] = (float x) => activeFunctions[0](x) * (1 - activeFunctions[0](x));// its derivative
                    break;
                default:// passing rectifier
                    activeFunctions[0] = ReLu;// x for x>0, 0 otherwise
                    activeFunctions[1] = StepFunction;// 1 for x>0;
                    break;
            }
            return activeFunctions;
        }

        private static float Tanh( float input)//  hyperbolic tanges
        {
            return (float)Math.Tanh(input);
        }
        private static float ReLu(float input) // rectifier
        {
            return input > 0 ? input : 0;
        }
        private static float StepFunction(float input) // step function
        {
            return input > 0 ? 1 : 0;
        }
    }
}
