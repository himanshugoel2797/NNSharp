using NNSharp.ANN.NetworkBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class FCLayer : ILayer, IWeightInitializable
    {
        private readonly int k;
        private readonly int output_dpth;
        private int input_sz, input_dpth;

        public Matrix Weights;
        public Matrix Biases;

        [NonSerialized]
        private bool layerReset;

        [NonSerialized]
        public Matrix ResultMemory;

        [NonSerialized]
        public Matrix CurDeltaMemory;

        [NonSerialized]
        public Matrix WeightDelta;

        [NonSerialized]
        private Matrix BiasDelta;

        [NonSerialized]
        private Matrix PrevInput;

        public FCLayer(int k, int output_dpth)
        {
            this.k = k;
            this.output_dpth = output_dpth;
            this.layerReset = false;
        }
         
        public void Learn(IOptimizer optimizer)
        {
            optimizer.RegisterLayer(this, 1, input_sz * input_sz * input_dpth, k * k * output_dpth, 1, k * k * output_dpth);
            optimizer.OptimizeWeights(this, 0, Weights, WeightDelta);
            optimizer.OptimizeBiases(this, 0, Biases, BiasDelta);
        }

        public void ResetLayerError()
        {
            //Clear the biases and deltas
            layerReset = true;
            Matrix.Fmop(null, 0, null, 0, BiasDelta);
        }

        public Matrix[] Forward(Matrix[] input)
        {
            PrevInput = input[0];
            Matrix.Mad(Weights, input[0], Biases, ResultMemory, true);
            return new Matrix[] { ResultMemory };
        }

        public Matrix[] Propagate(Matrix[] prev_delta)
        {
            //Compute the error to propagate to the following layer
            Matrix.Mad(Weights.Transpose(), prev_delta[0], null, CurDeltaMemory, true);
            return new Matrix[] { CurDeltaMemory };
        }

        public Matrix[] GetLastDelta()
        {
            return new Matrix[] { CurDeltaMemory };
        }

        public void LayerError(Matrix[] prev_delta)
        {
            //Compute the current weights using prev_delta as the error
            Matrix.Mad(prev_delta[0], PrevInput, null, WeightDelta, layerReset);
            Matrix.Fmop(prev_delta[0], 1, null, 0, BiasDelta);

            layerReset = false;
        }

        #region Parameter Setup
        public int GetOutputSize()
        {
            return k;
        }

        public int GetOutputDepth()
        {
            return output_dpth;
        }

        public void SetInputSize(int sz, int dpth)
        {
            input_sz = sz;
            input_dpth = dpth;

            if (Weights == null) Weights = new Matrix(sz * sz * input_dpth, k * k * output_dpth, MemoryFlags.ReadWrite, false);
            if (Biases == null) Biases = new Matrix(k * k * output_dpth, 1, MemoryFlags.ReadWrite, false);

            BiasDelta = new Matrix(k * k * output_dpth, 1, MemoryFlags.ReadWrite, false);
            WeightDelta = new Matrix(sz * sz * input_dpth, k * k * output_dpth, MemoryFlags.ReadWrite, false);
            ResultMemory = new Matrix(k * k * output_dpth, 1, MemoryFlags.ReadWrite, false);
            CurDeltaMemory = new Matrix(sz * sz * input_dpth, 1, MemoryFlags.ReadWrite, false);
        }
        #endregion

        #region IWeightInitializable
        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] m_ws = new float[Weights.Rows];
            float[] b_ws = new float[Biases.Rows];

            //for (int j = 0; j < Weights.Width; j++)
            Parallel.For(0, Weights.Columns, (j) =>
            {
                for (int i = 0; i < Weights.Rows; i++)
                    m_ws[i] = (float)weightInitializer.GetWeight(Weights.Columns, Weights.Rows); //(i + j * Weights.Height + 1) / (Weights.Width * Weights.Height + 1); //

                Weights.Write(m_ws, j * Weights.Rows);
            }
            );

            Parallel.For(0, b_ws.Length, (i) =>
           {
               b_ws[i] = weightInitializer.GetBias();
           });

            Biases.Write(b_ws);
        }
        #endregion

        #region Static Factory
        public static LayerContainer Create(int output_side, int output_depth)
        {
            return new LayerContainer(new FCLayer(output_side, output_depth));
        }
        #endregion
    }
}
