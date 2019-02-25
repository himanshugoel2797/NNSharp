using NNSharp.ANN.Kernels;
using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class ConvLayer : ILayer, IWeightInitializable
    {
        private int inputSz = 0;
        private int outputSz = 0;
        private int inputDepth = 0;
        private readonly int filterSz = 0 /*F*/;
        private readonly int paddingSz = 0 /*P*/;
        private readonly int filterCnt = 0 /*K*/;
        private readonly int strideLen = 0 /*S*/;
        public Matrix[][] Weights;
        public Vector Bias;

        [NonSerialized]
        public Matrix[][] WeightErrors;

        [NonSerialized]
        private Vector BiasError;

        [NonSerialized]
        private Vector Output;

        [NonSerialized]
        private Vector PrevInput;

        [NonSerialized]
        private Vector BackwardDelta;

        const int SmallKernelRequirement = 400;

        public ConvLayer(int filter_side, int filter_cnt, int padding = 0, int stride = 1)
        {
            filterSz = filter_side;
            filterCnt = filter_cnt;
            paddingSz = padding;
            strideLen = stride;
        }

        public void ResetLayerError()
        {
            for (int i = 0; i < filterCnt; i++)
                for (int j = 0; j < inputDepth; j++)
                    Matrix.Mult(WeightErrors[i][j], 0);

            Vector.Mult(BiasError, 0);
        }

        private static void Convolve(float[] input, bool rotInput, int inputOff, int inputSz, int paddingSz, int strideLen, float[] filter, bool rotFilter, int filterOff, int filterSz, float[] output, bool rotOutput, int outputOff, int outputSz)
        {
            Parallel.For(0, outputSz, (y) =>
            {
                for (int x = 0; x < outputSz; x++)
                    for (int y0 = 0; y0 < filterSz; y0++)
                    {
                        int i_y = y * strideLen + (y0 - filterSz / 2) + filterSz / 2 - paddingSz;
                        if (i_y >= 0 && i_y < inputSz)
                            for (int x0 = 0; x0 < filterSz; x0++)
                            {
                                int i_x = x * strideLen + (x0 - filterSz / 2) + filterSz / 2 - paddingSz;

                                if (i_x >= 0 && i_x < inputSz)
                                {
                                    float filter_val = filter[filterOff + (filterSz - 1 - y0) * filterSz + (filterSz - 1 - x0)];
                                    if (rotFilter) filter_val = filter[filterOff + y0 * filterSz + x0];

                                    float input_val = input[inputOff + i_y * inputSz + i_x];
                                    if (rotInput) input_val = input[inputOff + (inputSz - 1 - i_y) * inputSz + (inputSz - 1 - i_x)];

                                    float output_val = filter_val * input_val;

                                    if (rotOutput) output[outputOff + (outputSz - 1 - y) * outputSz + (outputSz - 1 - x)] += output_val;
                                    else output[outputOff + y * outputSz + x] += output_val;
                                }
                            }
                    }
            });
        }

        public Vector[] Propagate(Vector[] prev_delta)
        {
            //cur_delta = BackwardDelta = Full convolution of prev_delta with 180 rotated filter <- sum from all filters in terms of filterCnt, but spread across inputDepth? 

            //Clear BackwardDelta
            Vector.Mult(BackwardDelta, 0);
            for (int j = 0; j < inputDepth; j++)
            {
                for (int i = 0; i < filterCnt; i++)
                {
                    //Add the Full convolution of prev_delta with 180 rotated current filter
                    //M = outputSz
                    //N = filterSz
                    //K = inputSz
                    //S = stride
                    //P = required padding
                    //OX = N / 2
                    //OY = N / 2
                    //OZ = i
                    //OW = j
                    //kern = Weights[i][j]
                    //i = prev_delta
                    //o = BackwardDelta
                    int pSz = ((inputSz - 1) * strideLen - filterSz + outputSz) / 2;
                    int padd = ((inputSz - 1) * strideLen - outputSz + filterSz) / 2;
#if GPU
                    KernelManager.Convolve(prev_delta[0], i * outputSz * outputSz, outputSz, Weights[i][j], 0, filterSz, true, padd, strideLen, BackwardDelta, j * inputSz * inputSz, inputSz, false);

#elif CPU
                    //Convolve(Weights[i][j].memory, false, 0, filterSz, outputSz - 1, strideLen, prev_delta.memory, true, i * outputSz * outputSz, outputSz, BackwardDelta.memory, true, j * inputSz * inputSz, inputSz);
                    Convolve(prev_delta[0].memory, false, i * outputSz * outputSz, outputSz, padd, strideLen, Weights[i][j].memory, true, 0, filterSz, BackwardDelta.memory, false, j * inputSz * inputSz, inputSz);
#endif
                }
            }

            return new Vector[] { BackwardDelta };
        }

        public Vector[] GetLastDelta()
        {
            return new Vector[] { BackwardDelta };
        }

        public void LayerError(Vector[] prev_delta)
        {
            //Filter weight errors = covolution of Input with prev_delta <- doesn't tell us about individual filters per input dimension -> For now, treat the error as the same for each filter per input dimension
            for (int i = 0; i < filterCnt; i++)
            {
                Vector.VectorSum(BiasError, i, prev_delta[0], i * outputSz * outputSz, outputSz);
                for (int j = 0; j < inputDepth; j++)
                {
                    //convolution of input with prev_delta subportion for current filterCnt index

                    //Perform convolution of input[inputSz * inputSz * j] from prev_delta[outputSz * outputSz * i] to get the error for WeightErrors[i][j]
                    //M = inputSz
                    //N = outputSz
                    //K = filterSz
                    //S = stride
                    //OX = N / 2
                    //OY = N / 2
                    //OZ = i
                    //OW = j
                    //kern = prev_delta
                    //i = input
                    //o = WeightErrors[i][j]

                    int padd = ((filterSz - 1) * strideLen - inputSz + outputSz) / 2;
#if GPU
                    KernelManager.Convolve(PrevInput, j * inputSz * inputSz, inputSz, prev_delta[0], i * outputSz * outputSz, outputSz, true, padd, strideLen, WeightErrors[i][j], 0, filterSz, true);
#elif CPU
                           //int oSz = (i - fSz + 2 * pSz) / strLen + 1;
                           //oSz = filterSz
                           //i = inputSz
                           //fSz = outputSz
                           //pSz = ??
                           //strLen = strideLen
                           //((filterSz - 1) * strideLen - inputSz + outputSz)/2

                        Convolve(PrevInput.memory, false, j * inputSz * inputSz, inputSz, padd, strideLen, prev_delta[0].memory, true, i * outputSz * outputSz, outputSz, WeightErrors[i][j].memory, true, 0, filterSz);
#endif
                }
            }
        }

        public Vector[] Forward(Vector[] input)
        {
            PrevInput = input[0];

            //Clear the output vector
            Vector.Mult(Output, 0);

            //Foreach Filter Count
            int origin_base_x = filterSz / 2, origin_base_y = filterSz / 2;
            for (int i = 0; i < filterCnt; i++)
            {
                //Foreach Input Layers
                for (int j = 0; j < inputDepth; j++)
                {
                    //Perform convolution and add to the associated filter
                    //M = inputSz
                    //N = filterSz
                    //K = outputSz
                    //S = stride
                    //P = padding
                    //OX = origin_base_x
                    //OY = origin_base_y
                    //OZ = i
                    //OW = j
                    //kern = Weights[i][j]
                    //i = input
                    //o = Output

#if GPU
                    KernelManager.Convolve(input[0], j * inputSz * inputSz, inputSz, Weights[i][j], 0, filterSz, false, paddingSz, strideLen, Output, i * outputSz * outputSz, outputSz, false);
#elif CPU
                    Convolve(input[0].memory, false, j * inputSz * inputSz, inputSz, paddingSz, strideLen, Weights[i][j].memory, false, 0, filterSz, Output.memory, false, i * outputSz * outputSz, outputSz);
#endif
                }
                Vector.Add(Output, Bias, i);
            }

            return new Vector[] { Output };
        }

        public void Learn(IOptimizer optimizer)
        {
            optimizer.RegisterLayer(this, filterCnt * inputDepth, filterSz, filterSz, 1, filterCnt);

            int q = 0;
            for (int i = 0; i < filterCnt; i++)
            {
                for (int j = 0; j < inputDepth; j++)
                {
                    optimizer.Optimize(this, q++, Weights[i][j], WeightErrors[i][j]);
                }
            }

            optimizer.Optimize(this, 0, Bias, BiasError);
        }

        #region Parameter Setup
        public int GetOutputSize()
        {
            return outputSz;
        }

        public int GetOutputDepth()
        {
            return filterCnt;
        }

        public void SetInputSize(int input_side, int input_depth)
        {
            inputDepth = input_depth;
            inputSz = input_side;
            outputSz = (inputSz - filterSz + 2 * paddingSz) / strideLen + 1;

            //Allocate memory for the filters
            //Weights = new Matrix(filterSz * filterSz * filterCnt, inputDepth, MemoryFlags.ReadWrite, false);
            //Each filter is filterSz x filterSz x inputDepth
            //The output for each point of the filter is the sum of the convolution of each individual filter
            Weights = new Matrix[filterCnt][];
            WeightErrors = new Matrix[filterCnt][];

            for (int i = 0; i < filterCnt; i++)
            {
                Weights[i] = new Matrix[inputDepth];
                WeightErrors[i] = new Matrix[inputDepth];
                for (int j = 0; j < inputDepth; j++)
                {
                    Weights[i][j] = new Matrix(filterSz, filterSz, MemoryFlags.ReadWrite, false);
                    WeightErrors[i][j] = new Matrix(filterSz, filterSz, MemoryFlags.ReadWrite, false);
                }
            }

            //Need intermediate storage for each filter
            Output = new Vector(outputSz * outputSz * filterCnt, MemoryFlags.ReadWrite, false);

            Bias = new Vector(filterCnt, MemoryFlags.ReadWrite, false);
            BiasError = new Vector(filterCnt, MemoryFlags.ReadWrite, false);

            BackwardDelta = new Vector(inputSz * inputSz * inputDepth, MemoryFlags.ReadWrite, false);
        }
        #endregion

        #region IWeightInitializable
        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] rng = new float[filterSz * filterSz];
            float[] rng_b = new float[filterCnt];
            for (int i = 0; i < filterCnt; i++)
            {
                for (int j = 0; j < inputDepth; j++)
                {
                    for (int x = 0; x < filterSz * filterSz; x++)
                    {
                        rng[x] = weightInitializer.GetWeight(inputSz * inputSz, outputSz * outputSz);
                    }

                    Weights[i][j].Write(rng);
                }
                rng_b[i] = weightInitializer.GetBias();
            }
            Bias.Write(rng_b);
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create(int filter_side, int filter_cnt, int padding = 0, int stride = 1)
        {
            var layer = new ConvLayer(filter_side, filter_cnt, padding, stride);
            return new LayerContainer(layer);
        }
        #endregion
    }
}
