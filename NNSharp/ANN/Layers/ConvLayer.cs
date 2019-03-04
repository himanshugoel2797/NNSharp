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
        private readonly float strideLen = 0 /*S*/;
        private readonly int dilation = 0 /*D*/;
        public Matrix[][] Weights;
        public Matrix Bias;

        [NonSerialized]
        public Matrix[][] WeightErrors;

        [NonSerialized]
        private Matrix BiasError;

        [NonSerialized]
        private Matrix Output;

        [NonSerialized]
        private Matrix PrevInput;

        [NonSerialized]
        private Matrix BackwardDelta;

        [NonSerialized]
        private bool WeightErrorsReset;

        const int SmallKernelRequirement = 400;

        public int InputSz { get => this.inputSz; private set => this.inputSz = value; }
        public int OutputSz { get => this.outputSz; private set => this.outputSz = value; }
        public int InputDepth { get => this.inputDepth; private set => this.inputDepth = value; }
        public int FilterSz => this.filterSz;
        public int PaddingSz => this.paddingSz;
        public int FilterCnt => this.filterCnt;
        public float StrideLen => this.strideLen;
        public int Dilation => this.dilation;

        public ConvLayer(int filter_side, int filter_cnt, int padding = 0, float stride = 1, int dilation = 1)
        {
            filterSz = filter_side;
            filterCnt = filter_cnt;
            paddingSz = padding;
            strideLen = stride;
            this.dilation = dilation;
            WeightErrorsReset = false;
        }

        public void ResetLayerError()
        {
            WeightErrorsReset = true;
        }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            BackwardDelta.Clear();
            Parallel.For(0, inputDepth, (j) =>
            {

                for (int i = 0; i < filterCnt; i++)
                {
                    int padd = (int)(((inputSz - 1) * strideLen - outputSz + filterSz * dilation) / 2);
                    Matrix.Convolve(prev_delta[0], false, i * outputSz * outputSz, outputSz, padd, dilation, strideLen, Weights[i][j], true, 0, filterSz, BackwardDelta, false, j * inputSz * inputSz, inputSz, false);
                }
            });

            return new Matrix[] { BackwardDelta };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { BackwardDelta };
        }

        public void LayerError(Matrix[] prev_delta)
        {
            Parallel.For(0, filterCnt, (i) =>
            {
                float acc = 0;
                for (int x = 0; x < outputSz * outputSz; x++)
                    acc += prev_delta[0].Memory[prev_delta[0].Index(i * outputSz * outputSz + x, 0)];

                if (WeightErrorsReset)
                    BiasError.Memory[BiasError.Index(i, 0)] = 0;

                BiasError.Memory[BiasError.Index(i, 0)] += acc;

                for (int j = 0; j < inputDepth; j++)
                {
                    int padd = (int)(((filterSz - 1) * strideLen - inputSz + outputSz * dilation) / 2);
                    Matrix.Convolve(PrevInput, false, j * inputSz * inputSz, inputSz, padd, dilation, strideLen, prev_delta[0], true, i * outputSz * outputSz, outputSz, WeightErrors[i][j], true, 0, filterSz, WeightErrorsReset);
                }
            });
            WeightErrorsReset = false;
        }

        public Matrix[] Forward(Matrix[] input)
        {
            PrevInput = input[0];

            Output.Clear();
            Parallel.For(0, filterCnt, (i) =>
            {
                //Foreach Input Layers
                for (int j = 0; j < inputDepth; j++)
                {
                    Matrix.Convolve(input[0], false, j * inputSz * inputSz, inputSz, paddingSz, dilation, strideLen, Weights[i][j], false, 0, filterSz, Output, false, i * outputSz * outputSz, outputSz, false, (j == 0 ? Bias : null), i);
                }
            });

            return new Matrix[] { Output };
        }

        public void Learn(IOptimizer optimizer)
        {
            optimizer.RegisterLayer(this, filterCnt * inputDepth, filterSz, filterSz, 1, filterCnt);

            //for (int i = 0; i < filterCnt; i++)
            Parallel.For(0, filterCnt, (i) =>
            {
                for (int j = 0; j < inputDepth; j++)
                {
                    optimizer.OptimizeWeights(this, i * inputDepth + j, Weights[i][j], WeightErrors[i][j]);
                }
            });

            optimizer.OptimizeBiases(this, 0, Bias, BiasError);
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
            outputSz = (int)((inputSz - filterSz * dilation + 2 * paddingSz) / strideLen + 1);
            //o = (i - f * d + 2 * p) / s + 1
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
            Output = new Matrix(outputSz * outputSz * filterCnt, 1, MemoryFlags.ReadWrite, false);

            Bias = new Matrix(filterCnt, 1, MemoryFlags.ReadWrite, false);
            BiasError = new Matrix(filterCnt, 1, MemoryFlags.ReadWrite, false);

            BackwardDelta = new Matrix(inputSz * inputSz * inputDepth, 1, MemoryFlags.ReadWrite, false);
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
                        rng[x] = weightInitializer.GetWeight(filterSz, filterSz) / filterCnt;
                    }

                    Weights[i][j].Write(rng);
                }
                rng_b[i] = weightInitializer.GetBias();
            }
            Bias.Write(rng_b);
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create(int filter_side, int filter_cnt, int padding = 0, float stride = 1, int dilation = 1)
        {
            var layer = new ConvLayer(filter_side, filter_cnt, padding, stride, dilation);
            return new LayerContainer(layer);
        }
        #endregion
    }
}
