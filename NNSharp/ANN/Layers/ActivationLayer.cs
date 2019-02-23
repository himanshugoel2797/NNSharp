using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class ActivationLayer : ILayer
    {
        public IActivationFunction ActivationFunction { get; private set; }
        private int inputSz = 0;

        [NonSerialized]
        public Vector Activation, PrevInput, DeltaActivation;

        public ActivationLayer(IActivationFunction func)
        {
            ActivationFunction = func;
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            //Hadamard of prev_delta with derivative of PrevInput
            Vector.HadamardAct(prev_delta, PrevInput, DeltaActivation, ActivationFunction.DerivActivation());
            return DeltaActivation;
        }

        public Vector Forward(Vector input)
        {
            PrevInput = input;
            //Run activation function
            Vector.Activation(input, Activation, ActivationFunction.Activation());
            return Activation;
        }

        public int GetOutputSize(int input)
        {
            return input;
        }

        public void Reset() { }

        public void Learn(IOptimizer optimizer)
        {

        }

        public void SetInputSize(int sz)
        {
            inputSz = sz;
            Activation = new Vector(sz, MemoryFlags.ReadWrite, false);
            DeltaActivation = new Vector(sz, MemoryFlags.ReadWrite, false);
        }
    }
}
