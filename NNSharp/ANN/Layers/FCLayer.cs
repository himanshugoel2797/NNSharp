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
        private int k;
        private int input_sz;

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

        public FCLayer(int k)
        {
            this.k = k;
        }

        public Vector Forward(Vector input)
        {
            PrevInput = input;
            Matrix.Madd(Weights, input, Biases, ResultMemory);
            return ResultMemory;
        }

        public void Learn(IOptimizer optimizer)
        {
            optimizer.Optimize(Weights, WeightDelta);
            optimizer.Optimize(Biases, BiasDelta);
        }

        public void Reset()
        {
            //Clear the biases and deltas
            Matrix.Mult(WeightDelta, 0);
            Vector.Mult(BiasDelta, 0);
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            if (update_cur)
            {
                //Compute the current weights using prev_delta as the error
                Matrix.MatrixProduct(prev_delta, PrevInput, WeightDelta);
                Vector.Add(prev_delta, BiasDelta);
            }

            //Compute the error to propagate to the following layer
            Matrix.TMmult(Weights, prev_delta, CurDeltaMemory);
            
            return CurDeltaMemory;
        }

        public int GetOutputSize(int input)
        {
            return k;
        }

        public void SetInputSize(int sz)
        {
            input_sz = sz;

            if (Weights == null) Weights = new Matrix(sz, k, MemoryFlags.ReadWrite, false);
            if (Biases == null) Biases = new Vector(k, MemoryFlags.ReadWrite, false);

            BiasDelta = new Vector(k, MemoryFlags.ReadWrite, false);
            WeightDelta = new Matrix(sz, k, MemoryFlags.ReadWrite, false);
            ResultMemory = new Vector(k, MemoryFlags.ReadWrite, false);
            CurDeltaMemory = new Vector(sz, MemoryFlags.ReadWrite, false);
        }

        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] m_ws = new float[Weights.Height];
            float[] b_ws = new float[Biases.Length];

            for (int j = 0; j < Weights.Width; j++)
            {
                for (int i = 0; i < Weights.Height; i++)
                    m_ws[i] = (float)weightInitializer.GetWeight(Weights.Width, Weights.Height); //(i + j * Weights.Height + 1) / (Weights.Width * Weights.Height + 1); //

                Weights.Write(m_ws, j * Weights.Height);
            }

            Parallel.For(0, b_ws.Length, (i) =>
           {
               b_ws[i] = weightInitializer.GetBias();
           });

            Biases.Write(b_ws);
        }
    }
}
