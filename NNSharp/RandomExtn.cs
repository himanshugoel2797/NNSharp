using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    public static class RandomExtn
    {
        public static double NextGaussian(this Random r, double mu = 0, double sigma = 1)
        {
            var rand_std_normal = Math.Sqrt(-2.0 * Math.Log(r.NextDouble())) * Math.Sin(2.0 * Math.PI * r.NextDouble());
            var rand_normal = mu + sigma * rand_std_normal;
            return rand_normal;
        }
    }
}
