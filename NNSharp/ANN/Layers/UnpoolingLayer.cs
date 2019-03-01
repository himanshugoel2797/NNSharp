using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class UnpoolingLayer : ILayer
    {
        private readonly int stride;
        private readonly int filter_side;
        private int input_depth;
        private int input_sz;
        private int output_sz;

        [NonSerialized]
        private Matrix BackwardError;

        [NonSerialized]
        private PoolingLayer Pool;

        [NonSerialized]
        private Matrix Output;

        internal UnpoolingLayer(int stride, int filter_side, PoolingLayer pool)
        {
            this.stride = stride;
            this.filter_side = filter_side;
            this.Pool = pool;
        }

        public Matrix[] Forward(Matrix[] input)
        {
            Output.Clear();

            var prev_d = input[0].Reshape(input_depth, input_sz * input_sz);
            for (int i = 0; i < input_depth; i++)
            {
                for (int row_o = 0; row_o < input_sz; row_o++)
                {
                    for (int col_o = 0; col_o < input_sz; col_o++)
                    {
                        for (int row_f = 0; row_f < filter_side; row_f++)
                            for (int col_f = 0; col_f < filter_side; col_f++)
                            {
                                int i_row = row_o * stride + (row_f - filter_side / 2) + filter_side / 2;
                                int i_col = col_o * stride + (col_f - filter_side / 2) + filter_side / 2;

                                Output.Memory[Output.Index(i, i_row * output_sz + i_col)] += Pool.PoolCache.Memory[Pool.PoolCache.Index(i, i_row * output_sz + i_col)] * prev_d.Memory[prev_d.Index(i, row_o * input_sz + col_o)];
                            }
                    }
                }
            }

            return new Matrix[] { Output.Reshape(input_depth * output_sz * output_sz, 1) };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { BackwardError.Reshape(input_depth * input_sz * input_sz, 1) };
        }

        public void LayerError(Matrix[] prev_delta) { }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            BackwardError.Clear();

            var prev_d = prev_delta[0].Reshape(input_depth, output_sz * output_sz);
            for (int i = 0; i < input_depth; i++)
            {
                for (int row_o = 0; row_o < input_sz; row_o++)
                {
                    for (int col_o = 0; col_o < input_sz; col_o++)
                    {
                        for (int row_f = 0; row_f < filter_side; row_f++)
                            for (int col_f = 0; col_f < filter_side; col_f++)
                            {
                                int i_row = row_o * stride + (row_f - filter_side / 2) + filter_side / 2;
                                int i_col = col_o * stride + (col_f - filter_side / 2) + filter_side / 2;

                                BackwardError.Memory[BackwardError.Index(i, row_o * input_sz + col_o)] += Pool.PoolCache.Memory[Pool.PoolCache.Index(i, i_row * output_sz + i_col)] * prev_d.Memory[prev_d.Index(i, i_row * output_sz + i_col)];
                            }
                    }
                }
            }

            return new Matrix[] { BackwardError.Reshape(input_depth * input_sz * input_sz, 1) };
        }

        public void Learn(IOptimizer opt)
        {
            //No parameters to learn
        }

        public void ResetLayerError()
        {
        }

        #region Parameter Setup
        public int GetOutputSize()
        {
            return output_sz;
        }

        public int GetOutputDepth()
        {
            return input_depth;
        }

        public void SetInputSize(int sz, int input_dpth)
        {
            input_sz = sz;
            input_depth = input_dpth;
            output_sz = filter_side + (input_sz - 1) * stride;

            Output = new Matrix(input_depth, output_sz * output_sz, MemoryFlags.ReadWrite, true);
            BackwardError = new Matrix(input_depth, input_sz * input_sz, MemoryFlags.ReadWrite, true);
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create(LayerContainer pooling)
        {
            if (pooling.CurrentLayer is PoolingLayer)
                return new LayerContainer(new UnpoolingLayer((pooling.CurrentLayer as PoolingLayer).stride, (pooling.CurrentLayer as PoolingLayer).filter_side, (pooling.CurrentLayer as PoolingLayer)));
            else
                throw new ArgumentException("Argument must be a PoolingLayer");
        }
        #endregion
    }
}
