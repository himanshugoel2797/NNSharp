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
        private readonly int inputSz;
        private readonly int inputDpth;

        [NonSerialized]
        private Matrix[] prevDelta;

        public InputLayer(int input_side, int input_depth)
        {
            inputSz = input_side;
            inputDpth = input_depth;
        }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            prevDelta = prev_delta;
            return prev_delta;
        }

        public Matrix[] GetLastDelta()
        {
            return prevDelta;
        }

        public void LayerError(Matrix[] prev_delta) { }

        public Matrix[] Forward(Matrix[] input)
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
