using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public class Softmax : ActivationFunctionBase
    {
        public Softmax() : base("softmax", "softmax_deriv") { }

        protected override string ActivationFunc()
        {
            return "activ_res = 1.0f / (1.0f + exp(-res));";
        }

        protected override string DerivActivationFunc()
        {
            return "const float res_a = 1.0f / (1.0f + exp(-res)); activ_res = res_a * (1.0f - res_a);";
        }
    }
}
