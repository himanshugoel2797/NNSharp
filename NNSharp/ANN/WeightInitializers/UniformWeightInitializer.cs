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
        private Random rng;
        private float b;

        public UniformWeightInitializer(int seed, float bias)
        {
            rng = new Random(seed);
            b = bias;
        }

        public float GetBias()
        {
            return (float)rng.NextGaussian(0, b);
        }

        public float GetWeight(int in_dim, int out_dim)
        {
            return (float)rng.NextGaussian(0, Math.Sqrt(6.0f / (in_dim + out_dim)));
        }
    }
}
