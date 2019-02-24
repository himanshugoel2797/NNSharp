using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class PoolingLayer : ICNNLayer
    {
        private int stride, filter_side, input_depth, input_sz, output_sz;

        [NonSerialized]
        private Vector CurOutput;

        [NonSerialized]
        private Vector PoolCache;

        [NonSerialized]
        private Vector BackwardError;

        [NonSerialized]
        private Kernel fwd_layer;
        
        [NonSerialized]
        private Kernel error_layer;

        public PoolingLayer(int stride, int filter_side, int inputDepth)
        {
            this.stride = stride;
            this.filter_side = filter_side;
            this.input_depth = inputDepth;
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            Vector.Mult(BackwardError, 0);
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
                    .SetArgumentMemory(prev_delta.memory)
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

                                BackwardError.memory[i * input_sz * input_sz + i_x * input_sz + i_y] += PoolCache.memory[i * input_sz * input_sz + i_x * input_sz + i_y] * prev_delta.memory[i * output_sz * output_sz + x * output_sz + y];
                            }
                    }
                }
#endif
            }

            return BackwardError;
        }

        public Vector Forward(Vector input)
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
                .SetArgumentMemory(input.memory)
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

                                float i_val = input.memory[i * input_sz * input_sz + i_x * input_sz + i_y];

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

            return CurOutput;
        }

        public int GetFlatOutputSize()
        {
            return output_sz;
        }

        public int GetInputDepth()
        {
            return input_depth;
        }

        public int GetOutputSize(int input)
        {
            int output_sz = 1 + (input - filter_side) / stride;
            return output_sz * output_sz * input_depth;
        }

        public void Learn(IOptimizer opt)
        {
            //No parameters to learn
        }

        public void Reset()
        {
        }

        public void SetInputSize(int sz)
        {
            input_sz = sz;
            output_sz = 1 + (sz - filter_side) / stride;

            CurOutput = new Vector(output_sz * output_sz * input_depth, MemoryFlags.ReadWrite, true);
            PoolCache = new Vector(input_sz * input_sz * input_depth, MemoryFlags.ReadWrite, true);
            BackwardError = new Vector(input_sz * input_sz * input_depth, MemoryFlags.ReadWrite, true);
        }
    }
}
