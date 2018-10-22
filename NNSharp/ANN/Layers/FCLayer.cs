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
        private IActivationFunction activation;
        private IOptimizer optimizer;

        private int input_sz;
        private int output_sz;

        private Matrix Weights;
        private Vector Biases;
        
        [NonSerialized]
        private Vector ResultMemory;

        [NonSerialized]
        private Vector ActivResultMemory;
        
        [NonSerialized]
        private Vector CurDeltaMemory;

        [NonSerialized]
        private Matrix WeightDelta;

        [NonSerialized]
        private Vector PrevInput;

        public FCLayer(int k, IActivationFunction func)
        {
            this.k = k;
            activation = func;
        }

        public Vector Forward(Vector input)
        {
            PrevInput = input;
            Matrix.MaddAct(Weights, input, Biases, ResultMemory, ActivResultMemory, activation.Activation());
            return ActivResultMemory;
        }

        public void Learn()
        {
            //Error(input, prev_delta, prev_w, out cur_delta, out cur_w);
            optimizer.Optimize(Weights, Biases, WeightDelta, CurDeltaMemory);
        }

        public void Error(Vector input, Vector prev_delta, Matrix prev_w, out Vector cur_delta, out Matrix cur_w)
        {
            cur_w = Weights;
            
            if (prev_w == null)
            {
                Vector.HadamardAct(prev_delta, ResultMemory, CurDeltaMemory, activation.DerivActivation());
                cur_delta = CurDeltaMemory;
            }
            else
            {
                Matrix.TMmultAct(prev_w, prev_delta, ResultMemory, CurDeltaMemory, activation.DerivActivation());
                cur_delta = CurDeltaMemory;
            }

            Matrix.MatrixProduct(cur_delta, PrevInput, WeightDelta);

            if (prev_w != null && float.IsNaN(prev_w.Read()[0]))
                throw new Exception();

            if (float.IsNaN(prev_delta.Read()[0]))
                throw new Exception();

            if (float.IsNaN(ResultMemory.Read()[0]))
                throw new Exception();

            if (float.IsNaN(CurDeltaMemory.Read()[0]))
                throw new Exception();

            if (float.IsNaN(WeightDelta.Read()[0]))
                throw new Exception();
        }

        public int GetOutputSize(int input)
        {
            return k;
        }

        public void SetInputSize(int sz)
        {
            input_sz = sz;
            output_sz = GetOutputSize(sz);

            if (Weights == null) Weights = new Matrix(sz, k, MemoryFlags.ReadWrite, false);
            WeightDelta = new Matrix(sz, k, MemoryFlags.ReadWrite, false);
            if (Biases == null) Biases = new Vector(k, MemoryFlags.ReadWrite, false);
            ResultMemory = new Vector(k, MemoryFlags.ReadWrite, false);
            ActivResultMemory = new Vector(k, MemoryFlags.ReadWrite, false);
            CurDeltaMemory = new Vector(k, MemoryFlags.ReadWrite, false);
        }

        public void SetWeights(IWeightInitializer weightInitializer)
        {
            float[] m_ws = new float[Weights.Height];
            float[] b_ws = new float[Biases.Length];

            for (int j = 0; j < Weights.Width; j++)
            {
                for (int i = 0; i < Weights.Height; i++)
                    m_ws[i] = weightInitializer.GetWeight(Weights.Width, Weights.Height);

                Weights.Write(m_ws, j * Weights.Height);
            }

            for (int i = 0; i < b_ws.Length; i++)
                b_ws[i] = weightInitializer.GetBias();
            
            Biases.Write(b_ws);
        }

        public void SetOptimizer(IOptimizer opt)
        {
            optimizer = opt;
        }

        public IOptimizer GetOptimizer()
        {
            return optimizer;
        }
    }
}
