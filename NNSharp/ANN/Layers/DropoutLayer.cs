using NNSharp.ANN.NetworkBuilder;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class DropoutLayer : ILayer
    {
        [NonSerialized]
        private Matrix mask;

        [NonSerialized]
        private Matrix output;

        [NonSerialized]
        private Matrix prevdelta;

        private readonly float P;
        private NRandom rng;
        private int input_sz, input_dpth;

        public bool Enabled { get; set; }

        public DropoutLayer(float p = 0.3f, int seed = 0)
        {
            P = p;
            Enabled = true;
            rng = new NRandom(seed);
        }

        public Matrix[] Forward(params Matrix[] input)
        {
            if (Enabled)
            {
                int len = input_sz * input_sz * input_dpth;
                //Parallel.For(0, len, (i) =>
                for(int i = 0; i < len; i++)
                {
                    mask.Memory[i] = (rng.NextDouble() > P) ? 1 : 0;
                    output.Memory[i] = mask.Memory[i] * input[0].Memory[i];
                }
                //);
                return new Matrix[] { output };
            }
            return input;
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { prevdelta };
        }

        public void LayerError(params Matrix[] prev_delta)
        {

        }

        public void Learn(IOptimizer opt)
        {

        }

        public Matrix[] Propagate(params Matrix[] prev_delta)
        {
            if (Enabled)
            {
                int len = input_sz * input_sz * input_dpth;
                //Parallel.For(0, len, (i) =>
                for (int i = 0; i < len; i++)
                {
                    prevdelta.Memory[i] = mask.Memory[i] * prev_delta[0].Memory[i];
                }
                //);
                return new Matrix[] { prevdelta };
            }
            return prev_delta;
        }

        public void ResetLayerError()
        {

        }

        public int GetOutputDepth()
        {
            return input_dpth;
        }

        public int GetOutputSize()
        {
            return input_sz;
        }

        public void SetInputSize(int input_side, int input_depth)
        {
            input_sz = input_side;
            input_dpth = input_depth;
            mask = new Matrix(input_sz * input_sz * input_dpth, 1, MemoryFlags.ReadWrite, true);
            output = new Matrix(input_sz * input_sz * input_dpth, 1, MemoryFlags.ReadWrite, true);
            prevdelta = new Matrix(input_sz * input_sz * input_dpth, 1, MemoryFlags.ReadWrite, true);
        }

        #region Static Factory
        public static LayerContainer Create(float p = 0.3f, int seed = 0)
        {
            var layer = new DropoutLayer(p, seed);
            return new LayerContainer(layer);
        }
        #endregion
    }
}
