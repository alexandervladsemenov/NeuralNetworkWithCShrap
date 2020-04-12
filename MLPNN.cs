using System;
using System.Linq;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    public delegate void ComputeGradient<T>( out object vaL, T outputExpected, T outputNN, bool grads);
    public delegate T ActivationFunction<T>(T m);
    public delegate T[] GetInputOutput<T>(int sampleSize, params int[] dimensions);// training data input outputs
    public delegate T GenerateEmptyMatrix<T>(int rows, int columns);
    public class MLPNN<T, U> : NNDataBase<T, U>, IMLPNN // T stands for a matrix format, U stands for a vector format
    {
        private U inputLayerSingle; // we use theme for validation. May be not here.
        private U outputLayerSingle;
        private T inputLayerSample;
        private T outputLayerSample;
        private T outputLayerSampleExpected;
        private List<T> neuronOutputValues;
        private List<T> neuronInputValues;
        private List<T> errorValues;
        private List<T> biasTempMatrix; // we construct a bias matrix form bias vectors
        int[] parameters; // input parameters
        private ActivationFunction<T>[] ActivationFunctions;
        private ActivationFunction<T>[] ActivationFunctionsOutputLayer;
        private int epochNumber; // number of iterations
        private double learningRate; // learning rate
        private double lTwoRegularization; // regularuzation
        int SampleSize { set; get; } // miniBatch size
        int BatchSize { set; get; } // the total size of the Batch
        int ValidSize { set; get; } // the total size of the Batch
        int[] SampleSet { set; get; } = null; // mini batch indexes array 
        ComputeGradient<T> ComputeGradient { set; get; }
        GetInputOutput<T> GetInputOutput { set; get; } // gets input output from the mini batch of the total batch
        GetInputOutput<T> GetVerfication { set; get; } //gets input/output from verification data set
        Func<T,T> NNInputOutput { set; get; } // here we formulate the rule how to convert float to integer in case of classification
        Func<T, T, object> ValidationFunction { set; get; } //
        GenerateEmptyMatrix<T> GenerateEmptyMatrix { set; get; }
        public MLPNN(TrainingDataMethodsAndParameters<T> trainingDataMethodsAndParameters,
            NetWorkMethodsAndParameters<T, U> netWorkMethodsAndParameters, params int[] parameters) : 
            base(netWorkMethodsAndParameters)
        {
            SampleSize = trainingDataMethodsAndParameters.sampleSize; // ini of sample size for gradien decent
            BatchSize = trainingDataMethodsAndParameters.batchSize;// ini the total batch size for gradient decent
            ValidSize = trainingDataMethodsAndParameters.validSize;// ini the verification batch  size for gradient decent
            ComputeGradient = trainingDataMethodsAndParameters.computeGradient;// comptutes gradient and the cost function
            GetInputOutput = trainingDataMethodsAndParameters.getInputOutput; // sets input output
            ActivationFunctions = trainingDataMethodsAndParameters.activationFunctions; //sets Activation function for hidden layer
            ActivationFunctionsOutputLayer = trainingDataMethodsAndParameters.activationFunctionsOutputLayer; //sets Activation function for hidden layer
            GenerateEmptyMatrix = trainingDataMethodsAndParameters.generateEmptyMatrix; // generates the Matrix
            epochNumber = trainingDataMethodsAndParameters.epochNumber; //pass epochnumber
            learningRate = trainingDataMethodsAndParameters.learningRate;// setting learning rate
            lTwoRegularization = trainingDataMethodsAndParameters.lTwoRegularization;// setting l2 regularization
            GetVerfication = trainingDataMethodsAndParameters.getVerfication;// verification function
            NNInputOutput = trainingDataMethodsAndParameters.nNInputOutput;// conveting float output into classification
            ValidationFunction = trainingDataMethodsAndParameters.validationFunction;// how we compare the true output with the computed output
            weightMatrix = new List<T>(NetworkSize.Length - 1); // initilaex the weigts
            biasMatrix = new List<U>(NetworkSize.Length - 1);//// initilaex the bias
            biasTempMatrix = new List<T>(NetworkSize.Length - 1);// initiate bias layers Matrix
            errorValues = new List<T>(NetworkSize.Length - 1);//// inti errors
            neuronOutputValues = new List<T>(NetworkSize.Length - 1);// ini outputes
            neuronInputValues = new List<T>(NetworkSize.Length - 1); //ini inputs
            this.parameters = parameters; // optiontl parameters// Now not sure what is going to be here
            SetTheNetwrok(); // intilaizng 
        }
        public void AdjustNetWork()
        {
            for (int i = 0; i < NetworkSize.Length - 1; i++)
                UpdateNetwork(i); // Updating network
        }

        public void BackwardPropagation()
        {
            for (int i = NetworkSize.Length - 2; i >=0;  i--)
                DoStepBackward(i); // Feedforward Propagation
        }

        public void ComputeCost(int iEpoch)
        {
            throw new NotImplementedException();
        }

        public void ComputeOutput()
        {
            foreach (U bVec in biasMatrix) // needs to move to set up input
                biasTempMatrix.Add(VecToMat(SampleSize,bVec)); // ini bias matrix // Note: we update the vectors biasMatrix
            T[] buff = GetInputOutput(SampleSize, SampleSet); //GetInputOutput(SampleSize, NetworkSize[0], NetworkSize.Last());// get input and expectedTrue output
            inputLayerSample = buff[0]; // passing the values of input
            outputLayerSampleExpected = buff[1]; // passing the values of trueOutput
//            return;
            for (int i = 0; i < NetworkSize.Length - 1; i++)
                DoStepForward(i); // Feedforward Propagation
            outputLayerSample = neuronOutputValues[NetworkSize.Length - 2]; //storing output
            biasTempMatrix.Clear(); // clearing the list
        }

        public void CreateOutputFile()
        {
            throw new NotImplementedException();
        }

        public void DoStepBackward(int layerIndex)
        {
            object val;
            if (layerIndex == NetworkSize.Length - 2)
            {
                ComputeGradient(out val, outputLayerSampleExpected, outputLayerSample,true); // compute initial gradient //need to check both - passing object and matrix, tommorow
                errorValues[layerIndex] = MatMatMulPointWise((T)val, ActivationFunctionsOutputLayer[1](neuronInputValues[layerIndex]));//compute the error in the output layer // casting works
            }
            else
            {
                errorValues[layerIndex] = MatMatMulPointWise(MatMatMulTranspose(weightMatrix[layerIndex+1], errorValues[layerIndex+1]), ActivationFunctionsOutputLayer[1](neuronInputValues[layerIndex]));//compute the error in the hidden layer // also to check tommorow
            }
        }

        public void DoStepForward(int layerIndex)
        {
//            Console.WriteLine($"ForwardStep in #layer {layerIndex}");
            neuronInputValues[layerIndex] = MatMatSum
                (MatMatMul(weightMatrix[layerIndex], layerIndex == 0 ? inputLayerSample: neuronOutputValues[layerIndex-1]),
                biasTempMatrix[layerIndex]); // passing z-values            
            if (layerIndex < NetworkSize.Length - 2)
                neuronOutputValues[layerIndex] = ActivationFunctions[0](neuronInputValues[layerIndex]); // checking if it is the last ( output) layers
            else
                neuronOutputValues[layerIndex] = ActivationFunctionsOutputLayer[0](neuronInputValues[layerIndex]); // [0] stands for the function ,[1] for its dereivatives
                       
        }

        public void Optimization()
        {
            SetInput();// set up input
            int epcohCounter = 0;
            while (epcohCounter < epochNumber)
            {
                int pointsSampled = 0;
                while (pointsSampled <= BatchSize)
                {
                    ComputeOutput(); // compute output
                    BackwardPropagation();// doBackwardPropagation
                    AdjustNetWork(); // update network
                    pointsSampled += SampleSize;
                }
                Console.WriteLine($"Error is {ShowLoss()} in the epoch {epcohCounter+1}"); // testing
                epcohCounter++; //increment
                Console.WriteLine("Epoch {0} is completed", epcohCounter);
            }
        }

        public void SetInput()
        {
                inputLayerSingle=GenerateUnitVector(SampleSize); // stores singles
                if(parameters.ElementAtOrDefault(1) > 0)
                    SampleSet = Randomizer.ReturnRandomIntegerArray(SampleSize,BatchSize); // sample set
        }

        public void UpdateNetwork(int layerIndex) // we need to go through over all basis set. Read tommorow more about stochastic gradient decent
        {
            object multiplier =  - learningRate / SampleSize;
            if (layerIndex==0)
            {
                weightMatrix[layerIndex] =
                  MatMatSum(weightMatrix[layerIndex], 
                  MatScalarMult(MatMatMulSecondTranspose(errorValues[layerIndex],inputLayerSample), // adjusting weights for the input layers
                  multiplier));
            }
            else
            {
                weightMatrix[layerIndex] =
                  MatMatSum(weightMatrix[layerIndex],
                  MatScalarMult(MatMatMulSecondTranspose(errorValues[layerIndex], neuronOutputValues[layerIndex-1]),
                  multiplier)); //  adjusting weights for the hidden and output layers
            }
            outputLayerSingle = MatVecMul(errorValues[layerIndex], inputLayerSingle); // sum all over samples
            biasMatrix[layerIndex] =
             PointWiseVecVecSum(biasMatrix[layerIndex],
                VectScalarMult(outputLayerSingle, multiplier) // need to think about it
                );
            //            throw new NotImplementedException();
        }
        public void SetTheNetwrok()
        {
            for (int i = 0; i < NetworkSize.Length - 1; i ++)
            {
                weightMatrix.Add(GenerateWeights(NetworkSize[i + 1], NetworkSize[i], parameters)); // generating weights
                biasMatrix.Add(GenerateBias(NetworkSize[i+1], parameters));// generating biases
                errorValues.Add(GenerateEmptyMatrix(NetworkSize[i + 1],SampleSize)); // setting zeroes for errors, memory allocation
                neuronOutputValues.Add(GenerateEmptyMatrix(NetworkSize[i + 1], SampleSize)); // setteing zeroes for output (a-values), memory allocation
                neuronInputValues.Add(GenerateEmptyMatrix(NetworkSize[i + 1], SampleSize));//setting zeroes for input (z-values), memory allocation
            }
            inputLayerSample = GenerateEmptyMatrix(NetworkSize[0], SampleSize); // setting zeroes for input layer, memory allocation
            outputLayerSample = GenerateEmptyMatrix( NetworkSize.Last(), SampleSize); // setting zeroes for output layer, memory allocation
        }
        // This is just for debugging
        public float Accuracy(params int[] inputParameters) // need to upload verification data set
        {
            int totalNumberOfIterations = inputParameters[0];
            int sizeOfVerification = inputParameters[1];
            if (totalNumberOfIterations == 0)
                totalNumberOfIterations = ValidSize;
            int verSetIndex = 0;
            int correctPredictions = 0;
            neuronOutputValues.Clear();
            neuronInputValues.Clear();
            for (int i = 0; i < NetworkSize.Length - 1; i++)
            {
                neuronOutputValues.Add(GenerateEmptyMatrix(NetworkSize[i + 1], sizeOfVerification)); // setteing zeroes for output (a-values), memory allocation
                neuronInputValues.Add(GenerateEmptyMatrix(NetworkSize[i + 1], sizeOfVerification));//setting zeroes for input (z-values), memory allocation
            }
            while (verSetIndex < totalNumberOfIterations)
            {
                foreach (U bVec in biasMatrix) // needs to move to set up input
                    biasTempMatrix.Add(VecToMat(sizeOfVerification, bVec)); // ini bias matrix // Note: we update the vectors biasMatrix
                T[] verBuff = GetVerfication(verSetIndex);
                inputLayerSample = verBuff[0]; // passing the values of input
                outputLayerSampleExpected = verBuff[1]; // passing the values of trueOutpu
                for (int i = 0; i < NetworkSize.Length - 1; i++)
                    DoStepForward(i); // Feedforward Propagation
                outputLayerSample = NNInputOutput(neuronOutputValues[NetworkSize.Length - 2]); //storing output
                correctPredictions += (int)ValidationFunction(outputLayerSampleExpected, outputLayerSample); // for classification, gives 0 or 1
                verSetIndex++;
                biasTempMatrix.Clear(); // clearing the list
            }
            return (float)correctPredictions/ totalNumberOfIterations * 100;
        }
        public T ShowInputLayer()
        {
            return inputLayerSample;
        }
        public T ShowOutputLayer()
        {
            return outputLayerSampleExpected;
        }
        public T ShowComputedOutputLayer()
        {
            return outputLayerSample;
        }
        public T ShowComputedError(int layerIndex)
        {
            return errorValues[layerIndex];
        }
        public T ShowWeights(int layerIndex)
        {
            return weightMatrix[layerIndex];
        }
        public U ShowBiases( int layerIndex)
        {
            return biasMatrix[layerIndex];
        }
        public double ShowLoss()
        {
            ComputeGradient(out object val, outputLayerSampleExpected, outputLayerSample, false);
            return (double)val;
        }
        public T ShowDelta()
        {
            object val;
            ComputeGradient(out val, outputLayerSampleExpected, outputLayerSample, true);
            return (T)val;
        }
    }
    public class TrainingDataMethodsAndParameters<T>
    {
        public ComputeGradient<T> computeGradient; // compute gradient
        public GetInputOutput<T> getInputOutput; // get input - expected output
        public GetInputOutput<T> getVerfication; // get input - expected output
        public Func<T, T> nNInputOutput; // here we formulate the rule how to convert float to integer in case of classification
        public Func<T, T, object> validationFunction; // validation rule
        public ActivationFunction<T>[] activationFunctions;// activation functions hidden layers
        public ActivationFunction<T>[] activationFunctionsOutputLayer;// activation functions output layers
        public GenerateEmptyMatrix<T> generateEmptyMatrix; // matrix generator
        public int batchSize; // batch size
        public int validSize; // batch size
        public int sampleSize;// sample size
        public int epochNumber; // epoch number//
        public double learningRate; // learning rate
        public double lTwoRegularization;// l2 regularization
    }
}
