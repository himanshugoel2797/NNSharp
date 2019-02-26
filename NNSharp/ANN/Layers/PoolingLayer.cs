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
                for(int x = 0; x < output_sz; x++)
                {
                    for (int y = 0; y < output_sz; y++)
                    {
                        for (int n0 = 0; n0 < filter_side; n0++)
                            for (int n1 = 0; n1 < filter_side; n1++)
                            {
                                int i_x = x * stride + (n0 - filter_side / 2) + filter_side / 2;
                                int i_y = y * stride + (n1 - filter_side / 2) + filter_side / 2;

                                BackwardError.memory[i * input_sz * input_sz + i_x * input_sz + i_y] += PoolCache.memory[i * input_sz * input_sz + i_x * input_sz + i_y] * prev_delta[0].memory[i * output_sz * output_sz + x * output_sz + y];
                            }
                    }
                }
#endif
            }

            return new Matrix[] { BackwardError };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { BackwardError };
        }

        public void LayerError(Matrix[] prev_delta) { }

        public Matrix[] Forward(Matrix[] input)
        {
#if GPU
            var dev = Device.GetDevice();

            if (fwd_layer == null)
                fwd_layer = dev.LoadKernel("fwd_maxpool", "", $"#define IN_D ({input_sz})", $"#define KERN_D ({filter_side})", $"#define OUT_D ({output_sz})", $"#define STRIDE ({stride})");
#endif

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
                for(int x = 0; x < output_sz; x++)
                {
                    for (int y = 0; y < output_sz; y++)
                    {
                        int off = 0;
                        float acc = float.MinValue;
                        for (int n0 = 0; n0 < filter_side; n0++)
                            for (int n1 = 0; n1 < filter_side; n1++)
                            {
                                int i_x = x * stride + (n0 - filter_side / 2) + filter_side / 2;
                                int i_y = y * stride + (n1 - filter_side / 2) + filter_side / 2;

                                float i_val = input[0].memory[i * input_sz * input_sz + i_x * input_sz + i_y];

                                PoolCache.memory[i * input_sz * input_sz + i_x * input_sz + i_y] = 0;
                                if (i_val > acc)
                                {
                                    off = i * input_sz * input_sz + i_x * input_sz + i_y;
                                    acc = i_val;
                                }
                            }

                        PoolCache.memory[off] = 1;
                        CurOutput.memory[i * output_sz * output_sz + x * output_sz + y] = acc;
                    }
                }
#endif
            }

            return new Matrix[] { CurOutput };
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

            CurOutput = new Matrix(output_sz * output_sz * input_depth, 1, MemoryFlags.ReadWrite, true);
            PoolCache = new Matrix(input_sz * input_sz * input_depth, 1, MemoryFlags.ReadWrite, true);
            BackwardError = new Matrix(input_sz * input_sz * input_depth, 1, MemoryFlags.ReadWrite, true);
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
