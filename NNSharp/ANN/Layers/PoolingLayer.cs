using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class PoolingLayer : ILayer
    {
        private readonly int stride;
        private readonly int filter_side;
        private int input_depth;
        private int input_sz;
        private int output_sz;

        [NonSerialized]
        private Matrix CurOutput;

        [NonSerialized]
        private Matrix PoolCache;

        [NonSerialized]
        private Matrix BackwardError;

#if GPU
        [NonSerialized]
        private Kernel fwd_layer;

        [NonSerialized]
        private Kernel error_layer;
#endif

        public PoolingLayer(int stride, int filter_side)
        {
            this.stride = stride;
            this.filter_side = filter_side;
        }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            BackwardError.Clear();
#if GPU
            var dev = Device.GetDevice();

            if (error_layer == null)
                error_layer = dev.LoadKernel("error_maxpool", "", $"#define IN_D ({input_sz})", $"#define KERN_D ({filter_side})", $"#define OUT_D ({output_sz})", $"#define STRIDE ({stride})");
#endif

            var prev_d = prev_delta[0].Reshape(input_depth, output_sz * output_sz);
            for (int i = 0; i < input_depth; i++)
            {
#if GPU
                error_layer
                    .SetArgument(i * input_sz * input_sz)
                    .SetArgument(i * output_sz * output_sz)
                    .SetArgumentMemory(prev_delta[0].memory)
                    .SetArgumentMemory(PoolCache.memory)
                    .SetArgumentMemory(BackwardError.memory);

                dev.Dispatch(error_layer, new uint[] { (uint)output_sz, (uint)output_sz }, null);
#elif CPU
                for(int row_o = 0; row_o < output_sz; row_o++)
                {
                    for (int col_o = 0; col_o < output_sz; col_o++)
                    {
                        for (int row_f = 0; row_f < filter_side; row_f++)
                            for (int col_f = 0; col_f < filter_side; col_f++)
                            {
                                int i_row = row_o * stride + (row_f - filter_side / 2) + filter_side / 2;
                                int i_col = col_o * stride + (col_f - filter_side / 2) + filter_side / 2;
                                
                                BackwardError.memory[BackwardError.Index(i, i_row * input_sz + i_col)] += PoolCache.memory[PoolCache.Index(i, i_row * input_sz + i_col)] * prev_d.memory[prev_d.Index(i, row_o * output_sz + col_o)];
                            }
                    }
                }
#endif
            }

            return new Matrix[] { BackwardError.Reshape(input_depth * input_sz * input_sz, 1) };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { BackwardError.Reshape(input_depth * input_sz * input_sz, 1) };
        }

        public void LayerError(Matrix[] prev_delta) { }

        public Matrix[] Forward(Matrix[] input)
        {
#if GPU
            var dev = Device.GetDevice();

            if (fwd_layer == null)
                fwd_layer = dev.LoadKernel("fwd_maxpool", "", $"#define IN_D ({input_sz})", $"#define KERN_D ({filter_side})", $"#define OUT_D ({output_sz})", $"#define STRIDE ({stride})");
#endif
            var a_input = input[0].Reshape(input_depth, input_sz * input_sz);
            for (int i = 0; i < input_depth; i++)
            {
#if GPU
                fwd_layer
                .SetArgument(i * input_sz * input_sz)
                .SetArgument(i * output_sz * output_sz)
                .SetArgumentMemory(input[0].memory)
                .SetArgumentMemory(PoolCache.memory)
                .SetArgumentMemory(CurOutput.memory);

                dev.Dispatch(fwd_layer, new uint[] { (uint)output_sz, (uint)output_sz }, null);
#elif CPU
                for(int row = 0; row < output_sz; row++)
                {
                    for (int col = 0; col < output_sz; col++)
                    {
                        int off = 0;
                        float acc = float.MinValue;
                        for (int n_row = 0; n_row < filter_side; n_row++)
                            for (int n_col = 0; n_col < filter_side; n_col++)
                            {
                                int i_row = row * stride + (n_row - filter_side / 2) + filter_side / 2;
                                int i_col = col * stride + (n_col - filter_side / 2) + filter_side / 2;

                                float i_val = a_input.memory[a_input.Index(i, i_row * input_sz + i_col)];

                                PoolCache.memory[PoolCache.Index(i, i_row * input_sz + i_col)] = 0;
                                if (i_val > acc)
                                {
                                    off = PoolCache.Index(i, i_row * input_sz + i_col);
                                    acc = i_val;
                                }
                            }

                        PoolCache.memory[off] = 1;
                        CurOutput.memory[CurOutput.Index(i, row * output_sz + col)] = acc;
                    }
                }
#endif
            }

            return new Matrix[] { CurOutput.Reshape(input_depth * output_sz * output_sz, 1) };
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
            output_sz = 1 + (sz - filter_side) / stride;

            CurOutput = new Matrix(input_depth, output_sz * output_sz, MemoryFlags.ReadWrite, true);
            PoolCache = new Matrix(input_depth, input_sz * input_sz, MemoryFlags.ReadWrite, true);
            BackwardError = new Matrix(input_depth, input_sz * input_sz, MemoryFlags.ReadWrite, true);
        }
#endregion

#region Static Factory
        public static LayerContainer Create(int stride, int filter_side)
        {
            return new LayerContainer(new PoolingLayer(stride, filter_side));
        }
#endregion
    }
}
