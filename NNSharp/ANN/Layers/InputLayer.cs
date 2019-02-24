using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class InputLayer : ILayer
    {
        private int inputSz, inputDpth;

        [NonSerialized]
        private Vector[] prevDelta;

        public InputLayer(int input_side, int input_depth)
        {
            inputSz = input_side;
            inputDpth = input_depth;
        }

        public Vector[] Propagate(Vector[] prev_delta)
        {
            prevDelta = prev_delta;
            return prev_delta;
        }

        public Vector[] GetLastDelta()
        {
            return prevDelta;
        }

        public void LayerError(Vector[] prev_delta) { }

        public Vector[] Forward(Vector[] input)
        {
            return input;
        }

        public void Learn(IOptimizer opt)
        {

        }

        public void ResetLayerError()
        {

        }

        #region Parameter Setup
        public int GetOutputSize()
        {
            return inputSz;
        }

        public int GetOutputDepth()
        {
            return inputDpth;
        }

        public void SetInputSize(int input_side, int input_depth)
        {
            inputSz = input_side;
            inputDpth = input_depth;
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create(int input_side, int input_depth)
        {
            return new LayerContainer(new InputLayer(input_side, input_depth));
        }
        #endregion
    }
}
