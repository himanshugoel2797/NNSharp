using NNSharp.ANN.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class DeConvLayer : ICNNLayer, IWeightInitializable
    {
        //Forward: Take an image as input and convolve it with the specified number of filters
        //Backward: Apply deconvolution to update the filters
        private int inputSz = 0, outputSz = 0, inputDepth = 0, filterSz = 0 /*F*/, paddingSz = 0 /*P*/, filterCnt = 0 /*K*/, strideLen = 0 /*S*/;
        public Matrix[][] Weights;

        [NonSerialized]
        public Matrix[][] WeightErrors;

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

        public DeConvLayer()
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

            Vector.Mult(BackwardDelta, 0);
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            if (update_cur)
            {
                //Filter weight errors = covolution of Input with prev_delta <- doesn't tell us about individual filters per input dimension -> For now, treat the error as the same for each filter per input dimension
                for (int i = 0; i < filterCnt; i++)
                {
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

                        int padd = ((filterSz - 1) * strideLen - outputSz + inputSz) / 2;
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
                        Parallel.For(0, filterSz, (x) =>
                          {
                              for (int y = 0; y < filterSz; y++)
                                  for (int n0 = 0; n0 < inputSz; n0++)
                                      for (int n1 = 0; n1 < inputSz; n1++)
                                      {
                                          int i_x = x * strideLen + (n0 - inputSz / 2) + inputSz / 2 - padd;
                                          int i_y = y * strideLen + (n1 - inputSz / 2) + inputSz / 2 - padd;

                                          if (i_x >= 0 && i_y >= 0 && i_x < outputSz && i_y < outputSz)
                                              WeightErrors[i][j].memory[x * filterSz + y] += PrevInput.memory[j * inputSz * inputSz + n0 * inputSz + n1] * prev_delta.memory[i * outputSz * outputSz + i_x * outputSz + i_y];
                                      }
                          });
#endif
                    }
                }
            }

            //cur_delta = BackwardDelta = Full convolution of prev_delta with 180 rotated filter <- sum from all filters in terms of filterCnt, but spread across inputDepth? 

            //Clear BackwardDelta
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
                    Parallel.For(0, inputSz, (x) =>
                      {
                          for (int y = 0; y < inputSz; y++)
                              for (int n0 = 0; n0 < filterSz; n0++)
                                  for (int n1 = 0; n1 < filterSz; n1++)
                                  {
                                      int i_x = x * strideLen + (n0 - filterSz / 2) + filterSz / 2 - padd;
                                      int i_y = y * strideLen + (n1 - filterSz / 2) + filterSz / 2 - padd;

                                      if (i_x >= 0 && i_y >= 0 && i_x < outputSz && i_y < outputSz)
                                          BackwardDelta.memory[j * inputSz * inputSz + x * inputSz + y] += Weights[i][j].memory[(filterSz - 1 - n0) * filterSz + (filterSz - 1 - n1)] * prev_delta.memory[i * outputSz * outputSz + i_x * outputSz + i_y];
                                  }
                      });
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

            int padd = inputSz - (int)Math.Sqrt(input.Length / inputDepth);
            int mat_bnds = (int)Math.Sqrt(input.Length / inputDepth) * (strideLen - 1) + 1;

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
                    KernelManager.Convolve(input, j * inputSz * inputSz, inputSz, Weights[i][j], 0, filterSz, false, paddingSz, strideLen, Output, i * outputSz * outputSz, outputSz);
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
                    Parallel.For(0, outputSz, (x) =>
                     {
                         for (int y = 0; y < outputSz; y++)
                             for (int n0 = 0; n0 < filterSz; n0++)
                                 for (int n1 = 0; n1 < filterSz; n1++)
                                 {
                                     int i_x = x + (n0 - filterSz / 2) + filterSz / 2 - padd;
                                     int i_y = y + (n1 - filterSz / 2) + filterSz / 2 - padd;

                                     if (i_x >= 0 && i_y >= 0 && i_x < mat_bnds && i_y < mat_bnds && i_x % strideLen == 0 && i_y % strideLen == 0)
                                         Output.memory[i * outputSz * outputSz + x * outputSz + y] += Weights[i][j].memory[n0 * filterSz + n1] * input.memory[j * inputSz * inputSz + i_x / strideLen * inputSz + i_y / strideLen];
                                 }
                     });
#endif
                }
            }
            return Output;
        }

        public void SetOutputSize(int outputSz, int filterSz)
        {
            this.outputSz = outputSz;
            this.filterSz = filterSz;
            inputSz = outputSz + filterSz - 1;
        }

        public int GetOutputSize(int input)
        {
            int outputSz = input - filterSz + 1;
            return outputSz * outputSz * filterCnt;
        }

        public void Learn(IOptimizer optimizer)
        {
            for (int i = 0; i < filterCnt; i++)
            {
                for (int j = 0; j < inputDepth; j++)
                {
                    optimizer.Optimize(Weights[i][j], WeightErrors[i][j]);
                }
            }
        }

        public int GetFlatOutputSize()
        {
            return outputSz;
        }

        public void SetStrideLength(int stride)
        {
            strideLen = stride;
        }

        public void SetFilterCount(int filter)
        {
            filterCnt = filter;
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
            BackwardDelta = new Vector(inputSz * inputSz * inputDepth, MemoryFlags.ReadWrite, false);
        }

        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] rng = new float[filterSz * filterSz];
            for (int i = 0; i < filterCnt; i++)
            {
                for (int j = 0; j < inputDepth; j++)
                {
                    for (int x = 0; x < filterSz * filterSz; x++)
                    {
                        rng[x] = weightInitializer.GetWeight(filterSz, filterSz);
                    }

                    Weights[i][j].Write(rng);
                }
            }
        }
    }
}
