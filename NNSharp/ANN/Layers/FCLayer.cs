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
        private int k, output_dpth;
        private int input_sz, input_dpth;

        public Matrix Weights;
        public Vector Biases;

        [NonSerialized]
        public Vector ResultMemory;

        [NonSerialized]
        public Vector CurDeltaMemory;

        [NonSerialized]
        public Matrix WeightDelta;

        [NonSerialized]
        private Vector BiasDelta;

        [NonSerialized]
        private Vector PrevInput;

        public FCLayer(int k, int output_dpth)
        {
            this.k = k;
            this.output_dpth = output_dpth;
        }
        
        public void Learn(IOptimizer optimizer)
        {
            optimizer.RegisterLayer(this, 1, input_sz * input_dpth, k * output_dpth, 1, k * output_dpth);
            optimizer.Optimize(this, 0, Weights, WeightDelta);
            optimizer.Optimize(this, 0, Biases, BiasDelta);
        }

        public void ResetLayerError()
        {
            //Clear the biases and deltas
            Matrix.Mult(WeightDelta, 0);
            Vector.Mult(BiasDelta, 0);
        }

        public Vector[] Forward(Vector[] input)
        {
            PrevInput = input[0];
            Matrix.Madd(Weights, input[0], Biases, ResultMemory);
            return new Vector[] { ResultMemory };
        }

        public Vector[] Propagate(Vector[] prev_delta)
        {
            //Compute the error to propagate to the following layer
            Matrix.TMmult(Weights, prev_delta[0], CurDeltaMemory);
            return new Vector[] { CurDeltaMemory };
        }

        public Vector[] GetLastDelta()
        {
            return new Vector[] { CurDeltaMemory };
        }

        public void LayerError(Vector[] prev_delta)
        {
            //Compute the current weights using prev_delta as the error
            Matrix.MatrixProduct(prev_delta[0], PrevInput, WeightDelta);
            Vector.Add(BiasDelta, prev_delta[0]);
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

            if (Weights == null) Weights = new Matrix(sz * input_dpth, k * output_dpth, MemoryFlags.ReadWrite, false);
            if (Biases == null) Biases = new Vector(k * output_dpth, MemoryFlags.ReadWrite, false);

            BiasDelta = new Vector(k * output_dpth, MemoryFlags.ReadWrite, false);
            WeightDelta = new Matrix(sz * input_dpth, k * output_dpth, MemoryFlags.ReadWrite, false);
            ResultMemory = new Vector(k * output_dpth, MemoryFlags.ReadWrite, false);
            CurDeltaMemory = new Vector(sz * input_dpth, MemoryFlags.ReadWrite, false);
        }
        #endregion

        #region IWeightInitializable
        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] m_ws = new float[Weights.Height];
            float[] b_ws = new float[Biases.Length];

            //for (int j = 0; j < Weights.Width; j++)
            Parallel.For(0, Weights.Width, (j) =>
            {
                for (int i = 0; i < Weights.Height; i++)
                    m_ws[i] = (float)weightInitializer.GetWeight(Weights.Width, Weights.Height); //(i + j * Weights.Height + 1) / (Weights.Width * Weights.Height + 1); //

                Weights.Write(m_ws, j * Weights.Height);
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
