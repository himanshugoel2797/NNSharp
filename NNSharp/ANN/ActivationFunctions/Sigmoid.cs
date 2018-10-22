using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public class Sigmoid : ActivationFunctionBase
    {
        public Sigmoid() : base("const float activ_res = 1.0f / (1.0f + exp(-res));", "const float res_a = 1.0f / (1.0f + exp(-res)); const float activ_res = res_a * (1.0f - res_a);") { }
    }
}
