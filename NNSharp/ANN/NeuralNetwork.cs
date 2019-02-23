using NNSharp.ANN.Layers;
using NNSharp.ANN.Optimizers;
using NNSharp.Tools;
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
        public int InputSize { get { return inputCnt; } }
        public int LayerCount { get { return layers.Count; } }
        public IList<ILayer> Layers { get { return layers; } }

        private IList<ILayer> layers;
        private int inputCnt;
        private ILossFunction loss;
        private Vector cost_deriv, cost;
        private float error = 0;
        private float[] costs;

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
            //if (input != inputCnt)
            //    throw new Exception("Invalid input size");

            int inV = input;
            for (int i = 0; i < LayerCount; i++)
            {
                var layer = layers[i];
                if (layer is FCLayer)
                {
                    inV = layer.GetOutputSize(inV);
                }
                else if (layer is ICNNLayer)
                {
                    if ((i > 0 && layers[i - 1] is ICNNLayer) || i == 0)
                    {
                        inV = layer.GetOutputSize((int)Math.Sqrt(inV / (layer as ICNNLayer).GetInputDepth()));
                    }
                    else
                    {
                        inV = layer.GetOutputSize((int)Math.Sqrt(inV / (layer as ICNNLayer).GetInputDepth()));
                    }
                }
                else if (layer is ActivationLayer)
                {
                    //same size
                }
            }

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

        public void TrainSingle(Vector input, Vector expectedOutput, IOptimizer optimizer)
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
            {
                cost_deriv = new Vector(y.Length, MemoryFlags.ReadWrite, false);
                cost = new Vector(y.Length, MemoryFlags.ReadWrite, false);
                costs = new float[y.Length];
            }

            Vector.Mult(cost_deriv, 0);
            Vector.Mult(cost, 0);

            loss.LossDeriv(y_hat, y, cost_deriv);
            loss.Loss(y_hat, y, cost);

            var prev_delta = cost_deriv;
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                layers[i].Reset();
                prev_delta = layers[i].Error(prev_delta, true);
            }

            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn(optimizer);
            }
        }

        public void TrainMultiple(Vector[] input, Vector[] expectedOutput, IOptimizer optimizer)
        {
            if (cost_deriv == null)
            {
                cost_deriv = new Vector(expectedOutput[0].Length, MemoryFlags.ReadWrite, false);
                cost = new Vector(expectedOutput[0].Length, MemoryFlags.ReadWrite, false);
                costs = new float[expectedOutput[0].Length];
            }

            Vector cost_deriv_tmp = new Vector(expectedOutput[0].Length, MemoryFlags.ReadWrite, true);
            Vector.Mult(cost_deriv, 0);
            Vector.Mult(cost, 0);

            Reset();

            for (int q = 0; q < input.Length; q++)
            {
                var networkVolume = new Vector[LayerCount + 1];
                networkVolume[0] = input[q];

                for (int i = 0; i < LayerCount; i++)
                {
                    networkVolume[i + 1] = layers[i].Forward(networkVolume[i]);
                }

                var y_hat = networkVolume.Last();
                var y = expectedOutput[q];

                Vector.Mult(cost_deriv_tmp, 0);
                loss.LossDeriv(y_hat, y, cost_deriv_tmp);
                loss.Loss(y_hat, y, cost);

                Vector.Add(cost_deriv_tmp, cost_deriv);

                var prev_delta = cost_deriv_tmp;
                for (int i = LayerCount - 1; i >= 0; i--)
                {
                    prev_delta = layers[i].Error(prev_delta, true);
                }
            }

            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn(optimizer);
            }

            cost_deriv_tmp.Dispose();
        }


        public void TrainSingle(Vector loss_deriv, IOptimizer optimizer)
        {
            var prev_delta = loss_deriv;
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                layers[i].Reset();
                prev_delta = layers[i].Error(prev_delta, true);
            }

            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn(optimizer);
            }
        }

        public void Reset()
        {
            for (int i = 0; i < layers.Count; i++)
                layers[i].Reset();
        }

        public float Error()
        {
            cost.Read(costs);
            float acc = 0;
            for (int i = 0; i < costs.Length; i++)
                acc += costs[i];
            error = acc;// / cost.Length;
            return error;
        }

        public void ErrorDerivVec(float[] costs)
        {
            cost_deriv.Read(costs);
        }

        public void ErrorVec(float[] costs)
        {
            cost.Read(costs);
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

        public void Learn(IOptimizer optimizer)
        {
            for (int i = 0; i < LayerCount; i++)
            {
                layers[i].Learn(optimizer);
            }
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                prev_delta = layers[i].Error(prev_delta, update_cur);
            }
            return prev_delta;
        }

        public float Check(float epsilon, bool printNorms)
        {
            NRandom rng = new NRandom(0);

            SGD sgd = new SGD();
            sgd.SetLearningRate(0);

            Vector rand_input = new Vector(InputSize, MemoryFlags.ReadWrite, true);
            float[] rand_input_vec = new float[InputSize];
            for (int i = 0; i < rand_input.Length; i++)
                rand_input_vec[i] = (i + 1) / (float)(rand_input.Length + 1); //(float)rng.NextDouble();
            rand_input.Write(rand_input_vec);

            Vector rand_output = new Vector(GetOutputSize(InputSize), MemoryFlags.ReadWrite, true);
            float[] rand_output_vec = new float[rand_output.Length];
            for (int i = 0; i < rand_output.Length; i++)
                rand_output_vec[i] = (i + 1) / (float)(rand_output.Length + 1); //(float)rng.NextDouble();
            rand_output.Write(rand_output_vec);

            float net_norm = 0;
            int norm_cnt = 0;

            for (int i = 0; i < layers.Count; i++)
            {

                if (layers[i] is Layers.FCLayer)
                {
                    var w_delta_mat = new float[(layers[i] as Layers.FCLayer).WeightDelta.Width * (layers[i] as Layers.FCLayer).WeightDelta.Height];
                    var delta_w = (layers[i] as Layers.FCLayer).WeightDelta.Width;
                    var delta_h = (layers[i] as Layers.FCLayer).WeightDelta.Height;

                    TrainSingle(rand_input, rand_output, sgd);
                    var w_delta_orig = new float[w_delta_mat.Length];
                    (layers[i] as Layers.FCLayer).WeightDelta.Read(w_delta_orig);

                    float diff_norm = 0;
                    float sum_norm = 0;
                    var net = new float[w_delta_mat.Length];

                    for (int j = 0; j < w_delta_mat.Length; j++)
                    {
                        //Update a weight by subtracting epsilon
                        var orig_weight = (layers[i] as Layers.FCLayer).Weights.Read()[j];
                        (layers[i] as Layers.FCLayer).Weights.Write(new float[] { orig_weight + epsilon }, j);
                        TrainSingle(rand_input, rand_output, sgd);
                        float net_loss0 = Error();

                        //Update a weight by adding epsilon
                        (layers[i] as Layers.FCLayer).Weights.Write(new float[] { orig_weight - epsilon }, j);
                        TrainSingle(rand_input, rand_output, sgd);
                        float net_loss1 = Error();

                        //Compute the derivative
                        w_delta_mat[j] = (net_loss0 - net_loss1) / (2 * epsilon);

                        //Reset the weight
                        (layers[i] as Layers.FCLayer).Weights.Write(new float[] { orig_weight }, j);

                        float diff = w_delta_mat[j] - w_delta_orig[j];
                        //if(diff < epsilon) Console.Clear();
                        //Console.WriteLine($"[{j}] (N - O)  {w_delta_mat[j]} - {w_delta_orig[j]} = {diff}");

                        net[j] = w_delta_mat[j] / (w_delta_orig[j] + float.Epsilon);
                        //if(net[j] > 1 + epsilon | net[j] < 1 - epsilon)Console.WriteLine($"[{j}] {net[j]}");

                        diff_norm += diff * diff;
                        sum_norm += (w_delta_mat[j] + w_delta_orig[j]) * (w_delta_mat[j] + w_delta_orig[j]);
                    }

                    diff_norm = (float)Math.Sqrt(diff_norm);
                    sum_norm = (float)Math.Sqrt(sum_norm);

                    net_norm += diff_norm / sum_norm;
                    norm_cnt++;

                    //Compare the derivative to the actual derivatives
                    if (printNorms) Console.WriteLine($"Norms: { diff_norm / sum_norm }");
                }


            }
            return net_norm / norm_cnt;
        }
    }
}
