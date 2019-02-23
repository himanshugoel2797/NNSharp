using NNSharp.ANN.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class ConvLayer : ICNNLayer, IWeightInitializable
    {
        //Forward: Take an image as input and convolve it with the specified number of filters
        //Backward: Apply deconvolution to update the filters
        private int inputSz = 0, outputSz = 0, inputDepth = 0, filterSz = 0 /*F*/, paddingSz = 0 /*P*/, filterCnt = 0 /*K*/, strideLen = 0 /*S*/;
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

#if GPU
        [NonSerialized]
        private Kernel bkwd;
        [NonSerialized]
        private Kernel bkwd_err;
        [NonSerialized]
        private Kernel fwd;

        [NonSerialized]
        private Kernel bkwd_s;
        [NonSerialized]
        private Kernel bkwd_err_s;
        [NonSerialized]
        private Kernel fwd_s;
#endif

        public ConvLayer()
        {
#if GPU
            /*var dev = Device.GetDevice();
            bkwd = dev.LoadKernel("conv_bkwd");
            bkwd_err = dev.LoadKernel("conv_bkwd_err");
            fwd = dev.LoadKernel("conv_fwd");
            bkwd_s = dev.LoadKernel("conv_bkwd_small");
            bkwd_err_s = dev.LoadKernel("conv_bkwd_err_small");
            fwd_s = dev.LoadKernel("conv_fwd_small");*/
#endif
        }

        public void Reset()
        {
            for (int i = 0; i < filterCnt; i++)
                for (int j = 0; j < inputDepth; j++)
                    Matrix.Mult(WeightErrors[i][j], 0);

            Vector.Mult(BiasError, 0);
        }

        private static void Convolve(float[] input, bool rotInput, int inputOff, int inputSz, int paddingSz, int strideLen, float[] filter, bool rotFilter, int filterOff, int filterSz, float[] output, bool rotOutput, int outputOff, int outputSz)
        {
            Parallel.For(0, outputSz, (y) =>
            //for (int y = 0; y < outputSz; y++)
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

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            if (update_cur)
            {
                //Vector.Add(BiasError, prev_delta);

                //Filter weight errors = covolution of Input with prev_delta <- doesn't tell us about individual filters per input dimension -> For now, treat the error as the same for each filter per input dimension
                for (int i = 0; i < filterCnt; i++)
                {
                    Vector.VectorSum(BiasError, i, prev_delta, i * outputSz * outputSz, outputSz);
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
                    /*
                    var dev = Device.GetDevice();

                    var kern_name = ((inputSz * inputSz <= SmallKernelRequirement) ? bkwd_err_s : bkwd_err);
                    kern_name
                                    .SetArgument(outputSz)
                                    .SetArgument(inputSz)
                                    .SetArgument(filterSz)
                                    .SetArgument(strideLen)
                                    .SetArgument(padd)
                                    .SetArgument(outputSz / 2)
                                    .SetArgument(outputSz / 2)
                                    .SetArgument(i)
                                    .SetArgument(j)
                                    .SetArgumentMemory(PrevInput.memory)
                                    .SetArgumentMemory(prev_delta.memory)
                                    .SetArgumentMemory(WeightErrors[i][j].memory);

                    dev.Dispatch(kern_name, new uint[] { (uint)(filterSz * filterSz), 1 }, null);*/

                    KernelManager.Convolve(prev_delta, i * outputSz * outputSz, outputSz, PrevInput, j * inputSz * inputSz, inputSz, false, padd, strideLen, WeightErrors[i][j], 0, filterSz);
#elif CPU
                        //Parallel.For(0, filterSz, (y) =>
                        /*for (int y = 0; y < filterSz; y++)
                        {
                            for (int x = 0; x < filterSz; x++)
                                for (int y0 = 0; y0 < inputSz; y0++)
                                    for (int x0 = 0; x0 < inputSz; x0++)
                                    {
                                        int i_x = x * strideLen + (x0 - inputSz / 2) + inputSz / 2 - padd;
                                        int i_y = y * strideLen + (y0 - inputSz / 2) + inputSz / 2 - padd;

                                        if (i_x >= 0 && i_y >= 0 && i_x < outputSz && i_y < outputSz)
                                            WeightErrors[i][j].memory[(filterSz - 1 - y) * filterSz + (filterSz - 1 - x)] += PrevInput.memory[j * inputSz * inputSz + y0 * inputSz + x0] * prev_delta.memory[i * outputSz * outputSz + (i_y) * outputSz + (i_x)];
                                    }
                        }*///);
                           //int oSz = (i - fSz + 2 * pSz) / strLen + 1;
                           //oSz = filterSz
                           //i = inputSz
                           //fSz = outputSz
                           //pSz = ??
                           //strLen = strideLen
                           //((filterSz - 1) * strideLen - inputSz + outputSz)/2

                        Convolve(PrevInput.memory, false, j * inputSz * inputSz, inputSz, padd, strideLen, prev_delta.memory, true, i * outputSz * outputSz, outputSz, WeightErrors[i][j].memory, true, 0, filterSz);
#endif
                    }
                }
            }

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
                    /*
                    var dev = Device.GetDevice();

                    var kern = ((filterSz * filterSz <= SmallKernelRequirement) ? bkwd_s : bkwd);
                    kern
                                    .SetArgument(outputSz)
                                    .SetArgument(filterSz)
                                    .SetArgument(inputSz)
                                    .SetArgument(strideLen)
                                    .SetArgument(padd)
                                    .SetArgument(filterSz / 2)
                                    .SetArgument(filterSz / 2)
                                    .SetArgument(i)
                                    .SetArgument(j)
                                    .SetArgumentMemory(Weights[i][j].memory)
                                    .SetArgumentMemory(prev_delta.memory)
                                    .SetArgumentMemory(BackwardDelta.memory);

                    dev.Dispatch(kern, new uint[] { (uint)(inputSz * inputSz), 1 }, null);*/
                    KernelManager.Convolve(prev_delta, i * outputSz * outputSz, outputSz, Weights[i][j], 0, filterSz, true, padd, strideLen, BackwardDelta, j * inputSz * inputSz, inputSz);
                     
#elif CPU
                    //Parallel.For(0, inputSz, (y) =>
                    /*for (int y = 0; y < inputSz; y++)
                    {
                        for (int x = 0; x < inputSz; x++)
                            for (int y0 = -pSz; y0 < filterSz; y0++)
                                for (int x0 = -pSz; x0 < filterSz; x0++)
                                {
                                    if (y0 < 0 | x0 < 0) continue;
                                    //int oSz = (i - fSz + 2 * pSz) / strLen + 1;
                                    //oSz = inputSz
                                    //i = filterSz
                                    //fSz = outputSz
                                    //pSz = ??
                                    //strLen = strideLen
                                    //int pSz = ((inputSz - 1) * strideLen - filterSz + outputSz)/2
                                    int i_x = x * strideLen + (x0 - outputSz / 2) + outputSz / 2;// - padd;
                                    int i_y = y * strideLen + (y0 - outputSz / 2) + outputSz / 2;// - padd;
                                                                                                 //TODO: Fix, padding applies to weights
                                    if (i_x >= 0 && i_y >= 0 && i_x < outputSz && i_y < outputSz)
                                        BackwardDelta.memory[j * inputSz * inputSz + (y) * inputSz + (x)] += Weights[i][j].memory[(y0) * filterSz + (x0)] * prev_delta.memory[i * outputSz * outputSz + (outputSz - 1 - i_y) * outputSz + (outputSz - 1 - i_x)];
                                }
                    }*///);
                    //Convolve(Weights[i][j].memory, false, 0, filterSz, outputSz - 1, strideLen, prev_delta.memory, true, i * outputSz * outputSz, outputSz, BackwardDelta.memory, true, j * inputSz * inputSz, inputSz);
                    Convolve(prev_delta.memory, false, i * outputSz * outputSz, outputSz, padd, strideLen, Weights[i][j].memory, true, 0, filterSz, BackwardDelta.memory, false, j * inputSz * inputSz, inputSz);
#endif
                }
            }

            return BackwardDelta;
        }

        public Vector Forward(Vector input)
        {
            PrevInput = input;

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
                    KernelManager.Convolve(input, j * inputSz * inputSz, inputSz, Weights[i][j], 0, filterSz, true, paddingSz, strideLen, Output, i * outputSz * outputSz, outputSz);
                    /*var dev = Device.GetDevice();
                    var kern = ((filterSz * filterSz <= SmallKernelRequirement) ? fwd_s : fwd);
                    kern
                                    .SetArgument(inputSz)
                                    .SetArgument(filterSz)
                                    .SetArgument(outputSz)
                                    .SetArgument(strideLen)
                                    .SetArgument(paddingSz)
                                    .SetArgument(origin_base_x)
                                    .SetArgument(origin_base_y)
                                    .SetArgument(i)
                                    .SetArgument(j)
                                    .SetArgumentMemory(Weights[i][j].memory)
                                    .SetArgumentMemory(input.memory)
                                    .SetArgumentMemory(Output.memory);

                    dev.Dispatch(kern, new uint[] { (uint)(outputSz * outputSz), 1 }, null);
                    */
#elif CPU
                    //Dictionary<Tuple<int, int>, float> coords = new Dictionary<Tuple<int, int>, float>();
                    //Parallel.For(0, outputSz, (y) =>
                    //for (int x = 0; x < outputSz; x++)
                    /*{
                        for (int x = 0; x < outputSz; x++)
                            for (int y0 = 0; y0 < filterSz; y0++)
                                for (int x0 = 0; x0 < filterSz; x0++)
                                {
                                    int i_x = x * strideLen + (x0 - filterSz / 2) + filterSz / 2 - paddingSz;
                                    int i_y = y * strideLen + (y0 - filterSz / 2) + filterSz / 2 - paddingSz;


                                    if (i_x >= 0 && i_y >= 0 && i_x < inputSz && i_y < inputSz)
                                    {
                                        Output.memory[i * outputSz * outputSz + y * outputSz + x] += Weights[i][j].memory[(filterSz - 1 - y0) * filterSz + (filterSz - 1 - x0)] * input.memory[j * inputSz * inputSz + i_y * inputSz + i_x];
                                        //coords[new Tuple<int, int>(i_x + paddingSz, i_y + paddingSz)] = input.memory[j * inputSz * inputSz + i_x * inputSz + i_y];
                                    }
                                    else
                                    {
                                        //coords[new Tuple<int, int>(i_x + paddingSz, i_y + paddingSz)] = 0;
                                    }
                                }
                    });*/
                    /*
                    for (int y = 0; y < inputSz + 2 * paddingSz; y++)
                    {
                        Console.Write("[");
                        for (int x = 0; x < inputSz + 2 * paddingSz; x++)
                        {
                            Console.Write(" " + coords[new Tuple<int, int>(x, y)] + ".");
                        }
                        Console.WriteLine("]");
                    }*/
                    Convolve(input.memory, false, j * inputSz * inputSz, inputSz, paddingSz, strideLen, Weights[i][j].memory, false, 0, filterSz, Output.memory, false, i * outputSz * outputSz, outputSz);
#endif
                }
                //Console.WriteLine("\n\n");
                Vector.Add(Output, Bias, i);
            }

            return Output;
        }

        public int GetOutputSize(int input)
        {
            int outputSz = (input - filterSz + 2 * paddingSz) / strideLen + 1;
            return outputSz * outputSz * filterCnt;
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

        public int GetFlatOutputSize()
        {
            return outputSz;
        }

        public void SetStrideLength(int stride)
        {
            strideLen = stride;
        }

        public void SetFilterSize(int filter)
        {
            filterSz = filter;
        }

        public void SetFilterCount(int filter)
        {
            filterCnt = filter;
        }

        public void SetPaddingSize(int padding)
        {
            paddingSz = padding;
        }

        public void SetInputDepth(int d)
        {
            inputDepth = d;
        }

        public int GetInputDepth()
        {
            return inputDepth;
        }

        public void SetInputSize(int sz)
        {
            inputSz = sz;
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
    }
}
