using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.WeightInitializers
{
    [Serializable]
    public class UniformWeightInitializer : IWeightInitializer
    {
        private NRandom rng;
        private readonly float b;

        public UniformWeightInitializer(int seed, float bias)
        {
            rng = new NRandom(seed);
            b = bias;
        }

        public float GetBias()
        {
            return b;
        }

        public float GetWeight(int in_dim, int out_dim)
        {
            return (float)rng.NextGaussian(0, Math.Sqrt(2.0f / (in_dim + out_dim)));
        }
    }
}
