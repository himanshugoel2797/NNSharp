using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3
{
    public class Random
    {
        private ulong seed;

        private const ulong mult = 6364136223846793005;

        public Random(Random r)
        {
            this.seed = r.seed;
        }

        public Random() : this((int)(DateTime.Now.Ticks & 0x7fffffff))
        {

        }

        public Random(int seed)
        {
            this.seed = unchecked((ulong)seed);
        }

        public int Next()
        {
            seed = unchecked(seed * mult + 1);
            return unchecked((int)(seed >> 33));
        }

        public double NextDouble()
        {
            return unchecked(Next() % 20000) / (double)20000;
        }

        public double NextGaussian(double mu = 0, double sigma = 1)
        {
            var rand_std_normal = System.Math.Sqrt(-2.0 * System.Math.Log(NextDouble())) * System.Math.Sin(2.0 * System.Math.PI * NextDouble());
            var rand_normal = mu + sigma * rand_std_normal;
            return rand_normal;
        }
    }
}
