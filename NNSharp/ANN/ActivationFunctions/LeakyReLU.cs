using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public class LeakyReLU : ActivationFunctionBase
    {
        public const float Alpha = 0.3f;

        public LeakyReLU() : base("lrelu", "lrelu_deriv") {}

        protected override string ActivationFunc()
        {
            return "activ_res = isgreater(res, 0) * res + isless(res, 0) * 0.01f * res;";
        }

        protected override string DerivActivationFunc()
        {
            return "activ_res = isgreater(res, 0) + isless(res, 0) * 0.01f;";
        }
    }
}
