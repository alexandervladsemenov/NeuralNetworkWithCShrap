using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace CreatingTestCodeLibrary
{
    interface IMLPNN // set of methods for the MultiLayers Perceptons NeuralNetwork

    {
        void BackwardPropagation(); // Method Call Backward 
        void AdjustNetWork();
        void UpdateNetwork(int layerIndex); // Update Weights and Biases
        void Optimization();// optimization by calling Backpropagation algorithm
        void ComputeOutput(); // compute NN output
        void DoStepForward(int layerIndex); // Doing Forward propagation
        void DoStepBackward(int layerIndex); // STEP BACKWARD
        void SetInput(); // setting input
        void ComputeCost(int iEpoch);// computing cost for i-th Epochs
        void CreateOutputFile();// creating output file
        void SetTheNetwrok();// full ini of the Network
        float Accuracy(params int[] input); // returns how accurate the network
    }
}
