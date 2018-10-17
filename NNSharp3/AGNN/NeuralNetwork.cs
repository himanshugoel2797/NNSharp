using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.AGNN
{
    public class NeuralNetwork
    {
        List<LayerDef> layers;
        class LayerDef
        {
            public LayerType layerType;
            public ActivationFunction activation;
            public float[][] cmn;
            public float[][] p;
        }

        public int LayerCount { get { return layers.Count; } }
        public int InputSize { get { return layers[0].p[0].Length / layers[0].cmn[0].Length; } }
        public int OutputSize { get { return layers.Last().cmn[0].Length; } }

        public NeuralNetwork()
        {
            layers = new List<LayerDef>();
        }

        internal void Add(LayerType layerType, ActivationFunction activation, float[][] cmn, float[][] p)
        {
            layers.Add(new LayerDef()
            {
                layerType = layerType,
                activation = activation,
                cmn = cmn,
                p = p,
            });
        }

        internal void Add(NeuralNetwork nn)
        {
            layers.AddRange(nn.layers);
        }

        private float[] ForwardLayer(int idx, float[] input)
        {
            var lDef = layers[idx];

            int oLen = lDef.p[0].Length / input.Length;
            Parallel.For(0, oLen, (i) =>
            {
                float sum = 0;
                for (int j = 0; j < input.Length; j++)
                {
                    sum += input[j] + lDef.p[0][i * input.Length + j];
                }
                lDef.cmn[0][i] = sum;

                switch (lDef.activation)
                {
                    case ActivationFunction.Tanh:
                        lDef.cmn[1][i] = (float)System.Math.Tanh(sum);
                        break;
                    case ActivationFunction.Sigmoid:
                        lDef.cmn[1][i] = 1.0f / (1.0f + (float)System.Math.Exp(-sum));
                        break;
                    case ActivationFunction.ReLU:
                        lDef.cmn[1][i] = (sum > 0) ? sum : 0;
                        break;
                }
            });

            return lDef.cmn[1];
        }

        public float[] Forward(float[] input)
        {
            for (int l_idx = 0; l_idx < layers.Count; l_idx++)
            {
                input = ForwardLayer(l_idx, input);
            }
            return input;
        }

        public void ApplyGenome(Genome g)
        {
            Parallel.For(0, layers.Count, (i) =>
             {
                 for (int j = 0; j < g.NodeLen; j++)
                 {
                     Random rng = new Random(g.Nodes[i][j]);

                     for (int k = 0; k < layers[i].p[0].Length; k++)
                     {
                         layers[i].p[0][k] = (j == 0) ? (float)(2 * rng.NextDouble() - 1) : layers[i].p[0][k] + (float)rng.NextGaussian(0, 1f);//((2 * rng.NextDouble() - 1) * System.Math.Pow(2, -j));//rng.NextGaussian(0, System.Math.Pow(10 / System.Math.Min(j, 5), -System.Math.Min(j, 2)));

                         if (k < layers[i].p[1].Length)
                             layers[i].p[1][k] = (j == 0) ? (float)(2 * rng.NextDouble() - 1) : layers[i].p[1][k] + (float)rng.NextGaussian(0, 1f);// ((2 * rng.NextDouble() - 1) * System.Math.Pow(2, -j));//rng.NextGaussian(0, System.Math.Pow(10 / System.Math.Min(j, 5), -System.Math.Min(j, 2)));
                     }
                 }
             });
        }

        public float Loss(float[] expected_output)
        {
            var activ = layers.Last().cmn[1];

            float loss = 0;
            for (int i = 0; i < activ.Length; i++)
            {
                loss += 0.5f * (float)System.Math.Pow(activ[i] - expected_output[i], 2);
            }
            return loss / activ.Length;
        }
    }
}
