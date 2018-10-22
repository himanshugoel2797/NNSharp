using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    [Serializable]
    public class NeuralNetwork : ILayer
    {
        public int LayerCount { get { return layers.Count; } }

        private IList<ILayer> layers;
        private int inputCnt;
        private ILossFunction loss;

        private Vector cost_deriv;

        internal NeuralNetwork(IList<ILayer> layers, int inputCnt, ILossFunction loss)
        {
            this.inputCnt = inputCnt;
            this.layers = layers;
            this.loss = loss;
        }

        internal NeuralNetwork()
        { }

        public ILayer this[int idx]
        {
            get { return layers[idx]; }
        }

        public int GetOutputSize(int input)
        {
            if (input != inputCnt)
                throw new Exception("Invalid input size");

            int inV = input;
            for (int i = 0; i < LayerCount; i++)
                inV = layers[i].GetOutputSize(inV);

            return inV;
        }

        public void Save(string file)
        {
            var serializer = new BinaryFormatter();
            using (FileStream t = File.Create(file))
                serializer.Serialize(t, this);
        }

        public static NeuralNetwork Load(string file)
        {
            NeuralNetwork n = null;
            var serializer = new BinaryFormatter();

            using (FileStream t = File.OpenRead(file))
                n = (NeuralNetwork)serializer.Deserialize(t);

            int o_sz = 0;
            for (int i = 0; i < n.layers.Count; i++)
            {
                if (i > 0)
                {
                    n.layers[i].SetInputSize(o_sz);
                    o_sz = n.layers[i].GetOutputSize(o_sz);
                }
                else
                {
                    o_sz = n.layers[i].GetOutputSize(n.inputCnt);
                    n.layers[i].SetInputSize(n.inputCnt);
                }
            }

            return n;
        }

        public void SetInputSize(int sz)
        {

        }

        public void TrainSingle(Vector input, Vector expectedOutput)
        {
            var networkVolume = new Vector[LayerCount + 1];
            networkVolume[0] = input;


            for (int i = 0; i < LayerCount; i++)
            {
                networkVolume[i + 1] = layers[i].Forward(networkVolume[i]);
            }

            var y_hat = networkVolume.Last();
            var y = expectedOutput;

            if (cost_deriv == null)
                cost_deriv = new Vector(y.Length, MemoryFlags.ReadWrite, false);

            loss.LossDeriv(y_hat, y, cost_deriv);

            var prev_delta = cost_deriv;
            Matrix prev_w = null;
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                layers[i].Error(networkVolume[i], prev_delta, prev_w, out prev_delta, out prev_w);
            }

            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn();
            }
        }

        public Vector Forward(Vector input)
        {
            var inV = input;
            for (int i = 0; i < LayerCount; i++)
            {
                inV = layers[i].Forward(inV);
            }
            return inV;
        }

        public void SetOptimizer(IOptimizer opt)
        {

        }

        public IOptimizer GetOptimizer()
        {
            return null;
        }

        public void Learn()
        {
            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn();
            }
        }

        public void Error(Vector input, Vector prev_delta, Matrix prev_w, out Vector cur_delta, out Matrix cur_w)
        {
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                layers[i].Error(null, prev_delta, prev_w, out prev_delta, out prev_w);
            }

            cur_delta = prev_delta;
            cur_w = prev_w;
        }
    }
}
