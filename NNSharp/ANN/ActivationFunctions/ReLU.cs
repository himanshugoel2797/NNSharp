using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public class ReLU : ActivationFunctionBase
    {
        public ReLU() : base("relu", "relu_deriv") {}

        protected override string ActivationFunc()
        {
            return "activ_res = isgreater(res, 0) * res;";
        }

        protected override string DerivActivationFunc()
        {
            return "activ_res = isgreater(res, 0);";
        }
    }
}
