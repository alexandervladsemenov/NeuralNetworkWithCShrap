using System;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    public delegate T GenerateWeights<T>(int rows, int colums, params int[] parameters);
    public delegate U GenerateBias<U>(int rows, params int[] parameters);
    public delegate T MatMatMul<T>(T m1, T m2);
    public delegate T MatMatSum<T>(T m1, T m2);
    public delegate U MatVecMul<T,U>(T m, U v);
    public delegate U PointWiseVecVecSum<U>(U v1, U v2);
    public delegate T MatScalarMult<T>(T m, object scalar);
    public delegate U VectScalarMult<U>(U m, object scalar);
    public delegate T VecToMat<T, U>(int sampleSize, U v);
    public class NNDataBase<T, U> //generic class
    {
        protected List<T> weightMatrix;
        protected List<U> biasMatrix;
        protected int[] NetworkSize { set; get; }
        protected GenerateWeights<T> GenerateWeights { set; get; }
        protected GenerateBias<U> GenerateBias { set; get; }
        protected MatMatMul<T> MatMatMul { set; get; }
        protected MatMatMul<T>  MatMatMulPointWise { set; get; }
        protected MatMatMul<T> MatMatMulTranspose { set; get; }
        protected MatMatMul<T> MatMatMulSecondTranspose { set; get; }
        protected MatVecMul<T,U> MatVecMul { set; get; }
        protected MatMatSum<T> MatMatSum { set; get; }
        protected PointWiseVecVecSum<U> PointWiseVecVecSum { set; get; }
        protected VecToMat<T,U> VecToMat { set; get; }
        protected MatScalarMult<T> MatScalarMult { set; get; }
        protected VectScalarMult<U> VectScalarMult { set; get; }
        protected Func<int,U> GenerateUnitVector { set; get; }
        protected Func<T, T> TransposeMatrix { set; get; }
        // call cosntructor
        protected NNDataBase(NetWorkMethodsAndParameters<T, U> netWorkMethodsAndParameters) // make protected finally
        {
            GenerateWeights = netWorkMethodsAndParameters.generateWeights; // ini method for generating weights
            GenerateBias = netWorkMethodsAndParameters.generateBias;// ini method for generating bias
            MatMatMul = netWorkMethodsAndParameters.matMatMul;// ini method for Matrix x Matrix
            MatMatMulTranspose = netWorkMethodsAndParameters.matMatMulTranspose;// ini method for Matrix Transpose x Matrix
            MatMatMulSecondTranspose = netWorkMethodsAndParameters.matMatMulSecondTranspose;// ini method for Matrix x Matrix Transpose
            MatMatMulPointWise = netWorkMethodsAndParameters.matMatMulPointWise;// ini method for Matrix x Matrix PointWise
            MatVecMul = netWorkMethodsAndParameters.matVecMul;// ini method for Matrix x Vector
            MatMatSum = netWorkMethodsAndParameters.matMatSum; // ini method for Matrix + Matrix
            MatScalarMult = netWorkMethodsAndParameters.matScalarMult;// ini method mat x scalar
            VectScalarMult = netWorkMethodsAndParameters.vectScalarMult;// ini method vec x scalar
            PointWiseVecVecSum = netWorkMethodsAndParameters.pointWiseVecVecSum; // ini method for Shur product
            VecToMat = netWorkMethodsAndParameters.vecToMat; // ini method for construction of Matrix with Vector
            NetworkSize = netWorkMethodsAndParameters.networkSize;// number of layers, neurons
            GenerateUnitVector = netWorkMethodsAndParameters.GenerateUnitVector; // generate unit vector
            TransposeMatrix = netWorkMethodsAndParameters.TransposeMatrix; // matrix Transpose
        }
    }
    public class NetWorkMethodsAndParameters<T,U> // object which passes NN methds and parameters
    {
        public GenerateWeights<T> generateWeights; // stores GenWeights
        public GenerateBias<U> generateBias; // stores GenBias
        public MatMatMul<T> matMatMul; // stores MatMatMul
        public MatMatMul<T> matMatMulTranspose; // stores MatMatMul
        public MatMatMul<T> matMatMulSecondTranspose; // stores MatMatMul
        public MatMatMul<T> matMatMulPointWise; // stores MatMatMulPointWise
        public MatMatSum<T> matMatSum;// stores MatMatSum
        public MatVecMul<T, U> matVecMul; // stores MatVecMul
        public PointWiseVecVecSum<U> pointWiseVecVecSum;// stores ShurVecVecProduct
        public VecToMat<T,U> vecToMat;// Construct BiasToVec Matrix
        public MatScalarMult<T> matScalarMult; // mat x scalar product
        public VectScalarMult<U> vectScalarMult; // vec x scalar product
        public Func<int, U> GenerateUnitVector;// generate vectors filled with ones
        public Func<T, T> TransposeMatrix { set; get; } // transpose matrix
        public int[] networkSize; // NetWorkSize
    }
}
