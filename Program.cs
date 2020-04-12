using System;
using MathNet.Numerics;
using MathNet.Numerics.Random;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    class Program
    {
        static void Main(string[] args)
        {
            #region Setting up  binary data reader
            // ImageFileReader imageFileReader = new ImageFileReader(3 * 1024, 10); // COLOR IMAGES
            ImageFileReader imageFileReader = new ImageFileReader(28*28, 10,16,8); // MNIST DIGITS
            #endregion
            #region Initializng training Data
            int sampleSize = 50;// this is for test 10; // the Size of MiniBatch ( so called Sample)
            int inputRows = imageFileReader.Size; //10; // matrix sizes 
            int numberOfHiddenLayers = 1;// 1; // number of hidden layers
            int[] hiddenLayers = new int[numberOfHiddenLayers];
            for (int i = 0; i < numberOfHiddenLayers; i++)
                hiddenLayers[i] = inputRows/(int)Math.Pow(2,i+1)/3; // defining number of hidden layers
            int outputRows = imageFileReader.SizeOfClassification;// 1; // outputSizes // imageFileReader.SizeOfClassification
            int parameterToPass = 1; // if positive, random Normal generated weights and matrices, or unifrom otherwise
            int parameterSampleGenerator = 0; // if positive, sample is fixed// gradient decent
            int epochNumber = 20;// 300; // number of attempts
            double learningRate = 2; // learning rate 
            int[] NetWorkSize = new int[2 + numberOfHiddenLayers];// { rows, hiddenLayers outputRows }; // inti networkSize
            NetWorkSize[0] = inputRows; // ini input layer
            NetWorkSize[1 + numberOfHiddenLayers] = outputRows; // ini output layer
            for (int i = 1; i < 1 + numberOfHiddenLayers; i++)
                NetWorkSize[i] = hiddenLayers[i-1]; // ini of NN architecture
            NetWorkSize[1] = 30;//manual, comment after all
                #endregion
            #region SetUpTheNetWorkMethods
            NetWorkMethodsAndParameters<Matrix<float>,Vector<float>> parametersForNN = //iniParamtersMethod to Pass to a NN
                new NetWorkMethodsAndParameters<Matrix<float>, Vector<float>>();// creaty empty
            TrainingDataMethodsAndParameters<Matrix<float>> methodsForNN = //iniParamtersMethods to Pass to a Training
                new TrainingDataMethodsAndParameters<Matrix<float>>(); // create empty
            // ini for NN
            parametersForNN.generateWeights = RandomGenerators.GenerateRandomMatrix; // pass weights generators.
            parametersForNN.generateBias = RandomGenerators.GenerateRandomVector;// pass bias generators.
            parametersForNN.vecToMat = MathLibrary.VectorToMatrix; // vectorToMatrix for biases
            parametersForNN.matMatSum = MathLibrary.MatMatSum;// sum matrix and matrix
            parametersForNN.matMatMul = MathLibrary.MatMatMul; // product matrix and matrix
            parametersForNN.matMatMulTranspose = MathLibrary.MatMatMulTranspose; // product matrixT and matrix
            parametersForNN.matMatMulSecondTranspose = MathLibrary.MatMatMulSecondTranspose; // product matrix and matrixT
            parametersForNN.matMatMulPointWise = MathLibrary.MatMatMulPointWise; // product matrix and matrix, PointWise
            parametersForNN.matScalarMult =MathLibrary.MatScalarMul; // matrix x scalar product delegate
            parametersForNN.matVecMul = MathLibrary.MatVecMul;// matrix x vector;
            parametersForNN.vectScalarMult = MathLibrary.VecScalarMul; // matrix x scalar product delegate
            parametersForNN.pointWiseVecVecSum = MathLibrary.VectorVectorSum;// vect + vect;
            parametersForNN.GenerateUnitVector = MathLibrary.GenerateUnitVector;// generate unit vector
            parametersForNN.networkSize = NetWorkSize; // passint networkSize
            parametersForNN.TransposeMatrix = MathLibrary.TranposeMatrix;// Matrix Transpose
            // ini methods for propagation and feeding
            methodsForNN.sampleSize = sampleSize; // setting sampleSize
            methodsForNN.validSize = imageFileReader.ValidSize; // setting sampleSize
            methodsForNN.batchSize =  imageFileReader.BatchSize; // BatchSize for now disabled
            methodsForNN.epochNumber = epochNumber; // number of grad steps
            methodsForNN.learningRate = learningRate;// learning rate
            methodsForNN.getVerfication = imageFileReader.VerInOutSample;//computing verification
            methodsForNN.validationFunction = MathLibrary.ValidatePrediction;// validator
            methodsForNN.nNInputOutput = MathLibrary.GenerateMatrix;// generate input rule
            methodsForNN.generateEmptyMatrix = MathLibrary.GenerateMatrix; // null matrix generator
            methodsForNN.getInputOutput = imageFileReader.InputSample; // TrainingDataSet<Matrix<float>>.SetTrainingDataBenchmark; //setting InputOutput// imageFileReader.InputSample(sampleSize, null);
            methodsForNN.computeGradient =CostFunction.GradientQuadratic; // CostFunction.GradientEntropy;//  passing a gradient function // fix this issue tommorow
            ActivationFunctions.ReturnActivationFunction(out methodsForNN.activationFunctions,0, "Sigmoid", "Sigmoid"); // fixed // passing activation functions // hidden layers
            ActivationFunctions.ReturnActivationFunction(out methodsForNN.activationFunctionsOutputLayer,1, "Sigmoid", "Sigmoid"); // fixed // passing activation functions // output layers
            MLPNN<Matrix<float>, Vector<float>> myFullNN = new MLPNN<Matrix<float>, Vector<float>>(  methodsForNN, parametersForNN, parameterToPass, parameterSampleGenerator);// iniFullNetwork to Test
            #endregion
            #region Optimization and Output
            myFullNN.Optimization(); // running optimization procedure // printes cost function
            Console.WriteLine($"Accuracy is {myFullNN.Accuracy(10000,imageFileReader.SizeOfVerification)} %"); // ok it is working or it seems so.
            #endregion
            //// print Input
            //PrintMatrix(myFullNN.ShowInputLayer(), inputRows, sampleSize, "\nPrint InputLayer");
            //// print ComputedOutput 
            //PrintMatrix(myFullNN.ShowComputedOutputLayer(), outputRows, sampleSize, "\nPrint ComputedOutputLayer", 0);
            //// print TrueOutput
            //PrintMatrix(myFullNN.ShowOutputLayer(), outputRows, sampleSize, "\nPrint TrueOutputLayer");
            //// print RefinedComputedOutput
            //PrintMatrix(MathLibrary.GenerateMatrix(myFullNN.ShowComputedOutputLayer()), outputRows, sampleSize, "\nPrint ComputedOutputLayerModifiedByfunction");
            //Console.WriteLine($"The Number of CorrectAnswers is" +
            //    $" {(int)MathLibrary.ValidatePrediction(myFullNN.ShowOutputLayer(), MathLibrary.GenerateMatrix(myFullNN.ShowComputedOutputLayer()))}");

            //Console.WriteLine("\nPress Enter to Finish");
            //Console.ReadKey();
        }
        public static void PrintMatrix(Matrix<float> matrix, int rows, int columns, string name, params int[] outputProperty ) // printing matrix, testing
        {
            Console.WriteLine( name);
            for (int i = 0; i < columns; i++)
            {
                Console.Write("\n");
                for (int j = 0; j < rows; j++)
                {
                    var output = outputProperty.ElementAtOrDefault(0) > 0 ? (matrix[j, i]>0.5 ? 1:0 ) : matrix[j, i]; // rounding or not/ classification
                    Console.Write($"{output}, ");
                }
                    Console.Write(" end of the row;");
            }
            Console.Write("\n");
        }
    }
}
