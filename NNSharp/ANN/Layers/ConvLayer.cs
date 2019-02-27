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
        public Matrix Weights;
        public Matrix Bias;

        [NonSerialized]
        private Matrix PrevInputIC;

        [NonSerialized]
        public Matrix WeightErrors;

        [NonSerialized]
        private Matrix BiasError;

        [NonSerialized]
        private Matrix Output;

        [NonSerialized]
        private Matrix PrevInput;

        [NonSerialized]
        private Matrix BackwardDelta;

        [NonSerialized]
        private Matrix BackwardDeltaIC;

        [NonSerialized]
        private bool WeightErrorsReset;

        const int SmallKernelRequirement = 400;

        public ConvLayer(int filter_side, int filter_cnt, int padding = 0, int stride = 1)
        {
            filterSz = filter_side;
            filterCnt = filter_cnt;
            paddingSz = padding;
            strideLen = stride;
            WeightErrorsReset = false;
        }

        public void ResetLayerError()
        {
            WeightErrorsReset = true;
        }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            //cur_delta = BackwardDelta = Full convolution of prev_delta with 180 rotated filter <- sum from all filters in terms of filterCnt, but spread across inputDepth? 
            //Flatten and transpose the Weights as is, dot the flattened prev_delta
            Matrix.Mad(Weights.Transpose(), prev_delta[0].Reshape(filterCnt, outputSz * outputSz), null, null, BackwardDeltaIC, true);
            Matrix.Column2Image(inputSz, inputDepth, strideLen, paddingSz, filterSz, outputSz, BackwardDelta.Reshape(inputDepth, inputSz * inputSz), BackwardDeltaIC);
            //col2im the result
            return new Matrix[] { BackwardDelta };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { BackwardDelta };
        }

        public void LayerError(Matrix[] prev_delta)
        {
            //Filter weight errors = covolution of Input with prev_delta <- doesn't tell us about individual filters per input dimension -> For now, treat the error as the same for each filter per input dimension
            //Flatten prev_delta into a Vector, dot the transpose of the PrevInput
            var prev_delta_r = prev_delta[0].Reshape(filterCnt, outputSz * outputSz);
            Matrix.Mad(prev_delta_r, PrevInputIC.Transpose(), null, null, WeightErrors, WeightErrorsReset);
            //(filterCnt, outputSz * outputSz) . (outputSz * outputSz, filterSz * filterSz * inputDepth)

            for(int i = 0; i < filterCnt; i++)
            {
                float acc = 0;
                for (int x = 0; x < outputSz * outputSz; x++)
                    acc += prev_delta_r.memory[prev_delta_r.Index(i, x)];

                if (WeightErrorsReset)
                    BiasError.memory[BiasError.Index(i, 0)] = 0;

                BiasError.memory[BiasError.Index(i, 0)] += acc;
            }

            WeightErrorsReset = false;
        }

        public Matrix[] Forward(Matrix[] input)
        {
            PrevInput = input[0];

            Matrix.Image2Column(inputSz, inputDepth, strideLen, paddingSz, filterSz, outputSz, input[0].Reshape(inputDepth, inputSz * inputSz), PrevInputIC);
            Matrix.Mad(Weights, PrevInputIC, null, Bias, Output.Reshape(Weights.Rows, PrevInputIC.Columns), true);

            return new Matrix[] { Output };
        }

        public void Learn(IOptimizer optimizer)
        {
            optimizer.RegisterLayer(this, 1, Weights.Rows, Weights.Columns, 1, filterCnt);
            optimizer.OptimizeWeights(this, 0, Weights, WeightErrors);
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
            outputSz = (inputSz - filterSz + 2 * paddingSz) / strideLen + 1;

            //Allocate memory for the filters
            //Weights = new Matrix(filterSz * filterSz * filterCnt, inputDepth, MemoryFlags.ReadWrite, false);
            //Each filter is filterSz x filterSz x inputDepth
            //The output for each point of the filter is the sum of the convolution of each individual filter
            Weights = new Matrix(filterCnt, input_depth * filterSz * filterSz, MemoryFlags.ReadWrite, false);
            WeightErrors = new Matrix(filterCnt, input_depth * filterSz * filterSz, MemoryFlags.ReadWrite, false);

            //Need intermediate storage for each filter
            Output = new Matrix(outputSz * outputSz * filterCnt, 1, MemoryFlags.ReadWrite, false);

            Bias = new Matrix(filterCnt, 1, MemoryFlags.ReadWrite, false);
            BiasError = new Matrix(filterCnt, 1, MemoryFlags.ReadWrite, false);

            PrevInputIC = new Matrix(filterSz * filterSz * input_depth, outputSz * outputSz, MemoryFlags.ReadWrite, false);
            BackwardDeltaIC = new Matrix(filterSz * filterSz * input_depth, outputSz * outputSz, MemoryFlags.ReadWrite, false);
            BackwardDelta = new Matrix(inputSz * inputSz * inputDepth, 1, MemoryFlags.ReadWrite, false);
        }
        #endregion

        #region IWeightInitializable
        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] rng = new float[inputDepth * filterSz * filterSz];
            float[] rng_b = new float[filterCnt];

            for (int i = 0; i < filterCnt; i++)
            {
                int idx = 0;
                for (int j = 0; j < inputDepth; j++)
                    for (int x = 0; x < filterSz * filterSz; x++)
                        rng[idx++] = weightInitializer.GetWeight(inputSz * inputSz, outputSz * outputSz);
                rng_b[i] = weightInitializer.GetBias();

                Weights.Write(rng, i * inputDepth * filterSz * filterSz);
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
