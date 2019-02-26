using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class ActivationLayer : ILayer, IActivationLayer
    {
        public IActivationFunction ActivationFunction { get; private set; }
        private int inputSide = 0, inputDepth = 0, inputSz = 0;

        [NonSerialized]
        public Matrix Activation, PrevInput, DeltaActivation;

        public ActivationLayer(IActivationFunction func)
        {
            ActivationFunction = func;
        }

        public Matrix[] Propagate(params Matrix[] prev_delta)
        {
            if (prev_delta.Length != 1) throw new Exception();

            //Hadamard of prev_delta with derivative of PrevInput
            Matrix.HadamardActivation(PrevInput, prev_delta[0], DeltaActivation, ActivationFunction.DerivActivation());
            return new Matrix[] { DeltaActivation };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { DeltaActivation };
        }

        public void LayerError(params Matrix[] prev_delta)
        {

        }

        public Matrix[] Forward(params Matrix[] input)
        {
            if (input.Length != 1) throw new Exception();

            PrevInput = input[0];
            //Run activation function
            Matrix.HadamardActivation(input[0], null, Activation, ActivationFunction.Activation());
            return new Matrix[] { Activation };
        }

        public void ResetLayerError() { }

        public void Learn(IOptimizer optimizer) { }

        #region Parameter Setup
        public int GetOutputSize()
        {
            return inputSide;
        }

        public int GetOutputDepth()
        {
            return inputDepth;
        }

        public void SetInputSize(int input_side, int input_depth)
        {
            inputDepth = input_depth;
            inputSide = input_side;
            inputSz = input_side * input_side * input_depth;

            Activation = new Matrix(inputSz, 1, MemoryFlags.ReadWrite, false);
            DeltaActivation = new Matrix(inputSz, 1, MemoryFlags.ReadWrite, false);
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create<T>() where T : IActivationFunction, new()
        {
            T func = new T();
            return Create(func);
        }

        public static LayerContainer Create(IActivationFunction func)
        {
            return new LayerContainer(new ActivationLayer(func));
        }
        #endregion
    }
}
