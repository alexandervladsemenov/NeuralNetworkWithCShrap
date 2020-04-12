using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.Random;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    public delegate T GetOutputFromInput<T>(T inp, params int[] parameters); // delegate for outputFunction
    public delegate T GenerateInputSample<T>(params int[] parameters);// delegate for generating input
    public static class TrainingDataSet<T>
    {
        private static T inputSample;//inputSample
        private static T outputSampleExpected; // outputExpectedSample
        public static GetOutputFromInput<T> GetOutputFromInput { set; get; } // getting output from input
        public static GenerateInputSample<T> GenerateInputSample { set; get; } // getting input
        public static T[] SetTrainingDataBenchmark(int sampleSize = 1, params int[] dimensions)
        {
            // dimensions[0] - input size
            // dimensions[1] - output size
           inputSample = GenerateInputSample(dimensions[0],sampleSize); // getting input
           outputSampleExpected = GetOutputFromInput(inputSample, dimensions[0], sampleSize,dimensions[1]); // setting output for the input
           return new T[2] { inputSample, outputSampleExpected }; // give output back
        }
        public static void SetTrainingFromFiles(int sampleSize, int batchSize, params string[] fileNames)
        {

        }

        public static void SetTrainingFromDataBase(int sampleSize, int batchSize, params string[] fileNames)
        {

        }
     }
     static class Randomizer // generate random or selective input
    {
        public static WH2006 wH2006 = new WH2006(); // random number generator
        public static Func<int,int, float> ReturnFunction(float yMin, float yMax, bool rand, params int[] sizes  ) // returns a function 
        {
            if (rand)
                return (int rows, int columns) => (float)wH2006.NextDouble() * (yMax - yMin) + yMin; // random
            else
                return (int rows, int columns) =>  (yMax - yMin)/sizes[0]/sizes[1]*rows + columns* (yMax - yMin) / sizes[1] + yMin; // definiteve
        }

        public static Func<int, int, int[]> ReturnRandomIntegerArray =
            (int sampleSize, int maxVal) => wH2006.NextInt32s(sampleSize, 0, maxVal); //array of random numbers
        public static void Shuffle<T>(this WH2006 rng, T[] array) // shuflling data // Fisher - Yates Algorithms
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
        public static T[] SubArray<T>(this T[] data, int index, int length, Action<int> SetTheIndex) // SubArray. extenstion method
        {
            T[] result = new T[length];
            if (index + length <= data.Length)
            {
                Array.Copy(data, index, result, 0, length);
                SetTheIndex(index + length);
            }
            else
            {
                Array.Copy(data, data.Length - length, result, 0, length);
                SetTheIndex(data.Length);
            }
            return result;
        }
    }
    static class BenchmarkFunctions
    {
        private static float yMin = -4; //  min border for input
        private static float yMax = 4; //  maxs border for input
        private static float borderCoeff = 1; // dimension coeffienct
        private static float[] borders; // borders for Heavisie
        private static float[] values; // values 
        public static bool rand { set; get; } = false;
        static public void GenerateBordersAndValues(int size) // for Heaviside
        {
            borders = new float[size]; // borders Heaviside
            values = new float[size]; //values for Heavisides
            for (int i = 0; i < size; i++) 
            {
                values[i] = 1; // generating values
                borders[i] = i* borderCoeff;// generating borders
            }
        }
        static public Matrix<float> InputMatrixGenerate( params int[] parameters)
        {// generate somehow this matrix // tests for future

            return DenseMatrix.Create(parameters[0], parameters[1], Randomizer.ReturnFunction(yMin, yMax,rand, parameters[0], parameters[1]));
        }
        static public Matrix<float> ReturnMatrix(Matrix<float> inputMatrix, params int [] inputParameters)
        {// tests/ for fututre
            // inputParameters[0] - rows in the input matrix (input size) // input in NN
            // inputParameters[2] - rows in the output matrix ( output size) // SampleSize
            // inputParameters[1] - column size ( sample size) // output in NN
            float[] buff = new float[inputParameters[2]];// buffer of input parameters
            float[] input = new float[inputParameters[0]]; // it is read from input matrix
            float[,] inpArray = inputMatrix.ToArray(); // copying matrinx into an array // M[inputSize,sampleSize]
            float[,] outArray = new float[inputParameters[2], inputParameters[1]];// creating output array // M[outputSize,sampleSize]
            for (int i=0; i< inputParameters[1];i++) // loop over sample// Check
            {
                for(int j=0;j<inputParameters[0];j++) // loop over iput size
                input[j] = inpArray[j, i]; // copying into a buffer
                buff =
                    DoMath(input, borders, values, inputParameters[0], inputParameters[2]); // applying step function
                for (int j = 0; j < inputParameters[2]; j++) // loop over output size
                    outArray[j, i] = buff[j]; // copying back to the output /// need to debug here
            }
            return DenseMatrix.OfArray(outArray);//converting into a proper format
        }
        static public float[] DoMath(float[] input, float[] borders, float [] values, int size, int outputSize) // stepFunction
        {
            float result = 0;
            for (int i = 0; i < size; i++)
                if (input[i] > borders[i]) // Heavised on a border
                   result+=values[i] ;
            float[] output = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
                output[i] = result; // output evrywhere is the same
            return output;
        }
        static public float[] DoMath(float[] x, params int[] parameters) // sinFunction
        {
            float pi = (float)Math.Acos(-1); //pi number
            float[] output = new float[parameters[0]]; // output
            for (int i = 0; i < parameters[0]; i++) 
                for (int j = 0; j < parameters[1]; j++) // parameters[1] inputLength
                    output[i]+=(float)Math.Sin(pi*x[j]*(i+1)/parameters[0]); // initializing
            return output;
        }
    }

    public class ImageFileReader // Need to test MNIST database
    {
        public byte[] allData; // arrays for all data
        public byte[] verAllData; // arrays for all data
        public byte[] trainData; // training data
        public byte[] labels; // labels data
        public byte[] verificationData;// verification data
        public byte[] verificationLabels; // verification labels data
        public int Size { get; set; }// size of the input
        private int sizeOfTheData; // size of the data
        private int sizeOfLabels; // size of the total Batch
        private int verSizeOfTheData; // size of the data
        private int verSizeOfLabels; // size of the total Batch
        private int sampleVerCounter;//count nmuber of verifications
        private int[] allDataIndexes; // contains shuffelled indexes
        private int numberOfDataPointsPassed = 0;// number of data passed through
        private int SetNumberAndReshuffle //resets
        {
            set
            {
                numberOfDataPointsPassed = value;
                if (numberOfDataPointsPassed >= BatchSize)
                    numberOfDataPointsPassed = 0;
                if (numberOfDataPointsPassed == 0) // we reshuffle it every time
                {
                    Randomizer.wH2006.Shuffle(allDataIndexes);
                    Console.WriteLine("The Batch has been reshuffled");
                }// reshuffle
            }
            get
            {
                return numberOfDataPointsPassed;
            }
        }
        public int SizeOfVerification { set; get; } = 1; // size of verification
        private int SizeOfLabels //set size of lables
        {
            set
            {
                sizeOfLabels = value;
                BatchSize = sizeOfLabels;
            }
            get
            {
                return sizeOfLabels;
            }
        }
        private int sizeofTrainData; // byte size of train data
        private int offsetData = 0; // offsets for train data
        private int offsetLabels = 0;// offsets fpr labels
        private float maxValue = 1; // maxValue to normalize
        public int BatchSize { set; get; } // the total size of Bath
        public int ValidSize { set; get; } // the total size of Bath
        BinaryReader FileData; // reading for TrainData
        BinaryReader FileDataLabels; // reading for Data Labels
        public int SizeOfClassification { get; set; } // number of Classes
        public ImageFileReader(int size = 1, int sizeOfClassification = 1) //  CIFAR system
        {
            Size = size;
            SizeOfClassification = sizeOfClassification;
            string filePath = @"C:\Users\Alexander Semenov\Downloads\cifar-10-batches-bin\"; // file address
            string fileName = "data_batch_1.bin";
            FileData = new BinaryReader(new FileStream(string.Concat(filePath, fileName), FileMode.Open));
            sizeOfTheData = (int)FileData.BaseStream.Length;
            allData = FileData.ReadBytes(sizeOfTheData); // ok , it is RGBs
            SizeOfLabels = sizeOfTheData / (size + 1); // the size of the total batch
            sizeofTrainData = SizeOfLabels * size; // the size of infomration int the total batch
            labels = new byte[sizeOfLabels]; // create a label byte array
            trainData = new byte[sizeofTrainData]; // creata trainData
            for (int i = 0; i < sizeOfTheData; i++)
            {
                int labInd = i / (size + 1);
                if (i % (size + 1) == 0)
                    labels[i / (size + 1)] = allData[i];
                else
                    trainData[i - labInd - 1] = allData[i];
            }
            maxValue = allData.Max();
            Console.WriteLine($"Data have been read, {maxValue}");
            Console.WriteLine($"The total size of the data read is {sizeOfTheData}");
            Console.WriteLine($"The total size of the total batch is {SizeOfLabels}");
            Console.ReadKey();
        }
        public ImageFileReader(int size = 1, int sizeOfClassification = 1, int offsetData = 0, int offsetLabels = 0, params string[] namesPath) // MNIST DATA
        {
            //training data
            string filePathIdx = @"C:\Users\Alexander Semenov\Downloads\MNIST\train-images-idx3-ubyte\"; // training dataset inputs
            string fileNameIdx = "train-images.idx3-ubyte"; // input file
            string filePathIdxClass = @"C:\Users\Alexander Semenov\Downloads\MNIST\train-labels-idx1-ubyte\"; // training dataset outputs
            string fileNameIdxClass = "train-labels.idx1-ubyte"; //output file
            // verification data
            string filePathIdxVer = @"C:\Users\Alexander Semenov\Downloads\MNIST\t10k-images-idx3-ubyte\"; // training dataset inputs
            string fileNameIdxVer = "t10k-images.idx3-ubyte"; // input file
            string filePathIdxClassVer = @"C:\Users\Alexander Semenov\Downloads\MNIST\t10k-labels-idx1-ubyte\"; // training dataset outputs
            string fileNameIdxClassVer = "t10k-labels.idx1-ubyte"; //output file
            // parameters
            Size = size; // size of inputa data
            SizeOfClassification = sizeOfClassification; // number of classifications
            this.offsetData = offsetData; // offset for train data
            this.offsetLabels = offsetLabels; // offset for labels
            // reading training data
            FileData = new BinaryReader(new FileStream(string.Concat(filePathIdx, fileNameIdx), FileMode.Open));
            trainData = FileData.ReadBytes((int)FileData.BaseStream.Length); // reading train data
            FileDataLabels = new BinaryReader(new FileStream(string.Concat(filePathIdxClass, fileNameIdxClass), FileMode.Open));
            labels = FileDataLabels.ReadBytes((int)FileDataLabels.BaseStream.Length); // reading labels
            SizeOfLabels = (int)FileDataLabels.BaseStream.Length - offsetLabels; // size of the batch 
            sizeofTrainData = (int)FileData.BaseStream.Length - offsetData; // byte size of the train data
            sizeOfTheData = sizeofTrainData + SizeOfLabels; // labels + train data
            if ((size + 1) * SizeOfLabels != sizeOfTheData) // checking
                throw new Exception("Lables and Data do not match");
            allData = new byte[sizeOfTheData];
            for (int i = 0; i < SizeOfLabels; i++)
            {
                int startI = i * (size + 1); // labels
                int endI = (i + 1) * (size + 1);
                allData[startI] = labels[i + offsetLabels];
                for (int j = startI + 1; j < endI; j++)
                    allData[j] = trainData[j - 1 - i + offsetData]; // training data // here is the problem
            }
            maxValue = allData.Max();
            allDataIndexes = Enumerable.Range(0, sizeOfLabels).ToArray();// creating an aray of indexed
            SetNumberAndReshuffle = 0; // setting number for reshuffling and reshuflling array
            Console.WriteLine($"Idx File has been opened and its size is {FileData.BaseStream.Length}");
            Console.WriteLine($"Data have been read, max vlaue - {maxValue},\ndata size is {sizeOfTheData},\nnumber of labels is {SizeOfLabels} ");
            // reading verification data // to be continued
            //....
            FileData = new BinaryReader(new FileStream(string.Concat(filePathIdxVer, fileNameIdxVer), FileMode.Open));
            FileDataLabels = new BinaryReader(new FileStream(string.Concat(filePathIdxClassVer, fileNameIdxClassVer), FileMode.Open));
            verificationData = FileData.ReadBytes((int)FileData.BaseStream.Length); // reading verification data
            verificationLabels = FileDataLabels.ReadBytes((int)FileDataLabels.BaseStream.Length); // reading verification labels
            verSizeOfLabels = (int)FileDataLabels.BaseStream.Length - offsetLabels; // size of the verification labels
            ValidSize = verSizeOfLabels;// size of verification dataset
            verSizeOfTheData = (int)FileData.BaseStream.Length - offsetData; // byte size of the verification data
            if (size * verSizeOfLabels != verSizeOfTheData) // checking
                throw new Exception("Lables and Data do not match In the Verification DataSet");
            verAllData = new byte[verSizeOfTheData+ verSizeOfLabels];
            for (int i = 0; i < verSizeOfLabels; i++)
            {
                int startI = i * (size + 1); // labels
                int endI = (i + 1) * (size + 1);
                verAllData[startI] = verificationLabels[i + offsetLabels];
                for (int j = startI + 1; j < endI; j++)
                    verAllData[j] = verificationData[j - 1 - i + offsetData]; // training data // here is the problem
            }
            Console.WriteLine($"Tk 10 Idx File has been opened and its size is {FileData.BaseStream.Length}");
            Console.WriteLine($"Data Tk 10  have been read, data size is {verSizeOfTheData},\nnumber of labels is {verSizeOfLabels} ");
            Console.ReadKey();
        }
        public Matrix<float>[] InputSample(int sampleSize, int[] sampleIndex) // may need to re-write it
        {
            sampleSize = Math.Min(sampleSize, BatchSize);
            int[] sampleIndexToPass = new int[sampleSize];
            if (sampleIndex == null || sampleIndex.Length!= sampleSize)
                sampleIndexToPass = allDataIndexes.SubArray(SetNumberAndReshuffle, sampleSize, (int input) => SetNumberAndReshuffle = input);//new int[sampleSize]; // need to think how to initialze it porperly.
            else
                sampleIndexToPass = sampleIndex;
            Matrix<float>[] returnValues = new Matrix<float>[2]; // return training input{0} and ouput{1}
            Func<int, int, float> TakeData = (int r, int c) =>
                trainData[sampleIndexToPass[c] * Size + r + offsetData] / maxValue; //allData[sampleIndex[c] * (Size + 1) + r + 1]/maxValue; //reading all data from sampleIndex[c]th image, size = 1024x3 ( RGB), 0<=sampleIndex[c] <10000
            Func<int, int, float> TakeClass = (int r, int c) => {
                int x = labels[sampleIndexToPass[c] + offsetLabels];//allData[sampleIndex[c] * (Size + 1)]; // testing that the data read is correct
                if (x > SizeOfClassification - 1)
                    throw new Exception("Data Out Of Range");// c - index of a sample image, 0<=c<SampleSize, userDefined
                if (x == r) return 1; else return 0; // first symbol is a classifier, form 0 to  SizeOfClassification-1 
            };
            returnValues[1] = DenseMatrix.Create(SizeOfClassification, sampleIndexToPass.Length, TakeClass);// for now it is null
            returnValues[0] = DenseMatrix.Create(Size, sampleIndexToPass.Length, TakeData);  // for now it is null //seems working
            return returnValues;
        }
        public Matrix<float>[] VerInOutSample(int verSampleIndex, int [] parameters = null) // generates verification
        {
            Matrix<float>[] returnValues = new Matrix<float>[2];
            Func<int, int, float> TakeData = (int r,int c) =>
                verificationData[verSampleIndex * Size + r + offsetData] / maxValue;
            Func<int, int, float> TakeClass = (int r, int c) => {
                int x = verificationLabels[verSampleIndex + offsetLabels];//allData[sampleIndex[c] * (Size + 1)]; // testing that the data read is correct
                if (x > SizeOfClassification - 1)
                    throw new Exception("Data Out Of Range in Verification");// c - index of a sample image, 0<=c<SampleSize, userDefined
                if (x == r) return 1; else return 0; // first symbol is a classifier, form 0 to  SizeOfClassification-1 
            };
            returnValues[1] = DenseMatrix.Create(SizeOfClassification, 1, TakeClass);// for now it is null
            returnValues[0] = DenseMatrix.Create(Size, 1, TakeData);  // for now it is null //seems working
            return returnValues;
        }
    }

     public static class DefinedNetworkGenerators // sample size is just one for now
    {
        public static int outputSize { set; get; } = 1; //read from 
        public static int inputSize { set; get; } = 1;
        private static int sampleIndexCounter = 3;
        private static readonly float pi = 3.1415926f;
        private static readonly int minIntPass = 3;
        private static Func<int,int> max = (int i) => i> minIntPass ? i: minIntPass;
        private static Func<int, float> coeffOne = (int i)=> (float)(Math.Sqrt(1.0 / i) + Math.Sqrt(1.2 + i)); // generates coefficents 1
        private static Func<int, float> coeffTwo = (int i) => (float)(Math.Sqrt(1.0 / (i + 1.0)) + Math.Sqrt(1.5 + i));// generates coefficents 2
        private static Func<int, float> coeffThree = (int i) => (float)(Math.Sqrt(1.0 / (i + 2.5)) + Math.Sqrt(1.2 + i)); // generates coefficents 1
        private static Func<int, float> coeffFour = (int i) => (float)(Math.Sqrt(1.0 / (i + 1.7)) + Math.Sqrt(1.5 + i));// generates coefficents 2
        private static Func<int, int, int, float> Mapfunction = (int i, int j, int passI) => (float)(Math.Cos(pi * (j + 1) * coeffOne(passI)) // //mapping function for matrix
        + Math.Sin(pi * (i + 1) * coeffTwo(passI))); // i - rows, j - columns
        private static Func<int, int, float> InputGenerator = (int i, int passI)
            => (float)(Math.Exp(Math.Cos(pi*(i+1)*coeffThree(passI)))+Math.Log(Math.Abs(Math.Sin(pi*(i+1)*coeffFour(passI))))) ; //generator function
        private static Func<int, int, float> MatrixGenerator = (int i, int j) => (i + 1.0f) * (j + 1.0f) / 2.0f;
        private static Matrix<float> inpToOutMatrix = null;
        public static Matrix<float> GenerateWeights(int rows, int columns, params int[] indexGenerator) // generating weights
        {
            return  DenseMatrix.Create(rows, columns,(int i,int j)=> Mapfunction(i, j, max(indexGenerator[2])));
        }
        public static Vector<float> GenerateBias(int rows, params int[] indexGenerator)
        {
            return DenseVector.Create(rows, (int i) => Mapfunction(i-3, i, max(indexGenerator[2]))); // generate Bias vectors as well // do it in Python
        }
        public static Matrix<float>[] GenerateInputOutput(int sampleSize, int[] sampleIndex)
        {
            int[] sampleIndexToPass;
            if (sampleIndex == null)
            {
                sampleIndexToPass = Enumerable.Range(sampleIndexCounter, sampleIndexCounter + sampleSize-1).ToArray(); // we do not want to change sampleIndex
                sampleIndexCounter++;
            }
            else
            {
                sampleIndexToPass = sampleIndex;
            }
            inpToOutMatrix = DenseMatrix.Create(outputSize, inputSize, MatrixGenerator); // mult matrix
            Matrix<float> inpSample = DenseMatrix.Create(inputSize, sampleSize, (int i,int j) => InputGenerator(i, max(sampleIndexToPass[j]))); // output sample matrix mulpitplication
            Matrix<float> outSample = inpToOutMatrix.Multiply(inpSample);// M1[n1,n2] x M2[n2,n3] = M3[n1,n3]
            return new Matrix<float>[] { inpSample, outSample };
        }
    }
}
