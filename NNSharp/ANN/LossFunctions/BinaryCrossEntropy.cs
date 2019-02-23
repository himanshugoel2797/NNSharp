using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    [Serializable]
    public class BinaryCrossEntropy : LossFunctionBase
    {
#if CPU
        public BinaryCrossEntropy() : base("binary_cross_entropy", "binary_cross_entropy_deriv") { }
#elif GPU
        public BinaryCrossEntropy() : base(@"const float eps = 1e-12; activ_res = - (eo* log(o + eps) + (1-eo) * log(1 - o + eps));", "const float eps = 1e-12; \nactiv_res = 1.0f / (1.0f - eo - o); \nactiv_res *= (!isnan(activ_res));"/*activ_res = -(eo / (o + eps) - (1 - eo) / (1 - o + eps));"*/) { }
#endif
    }
}
