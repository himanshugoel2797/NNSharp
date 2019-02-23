using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public class Tanh : ActivationFunctionBase
    {
        public Tanh() : base("tanh", "tanh_deriv") { }

        protected override string ActivationFunc()
        {
            return "activ_res = tanh(res);";
        }

        protected override string DerivActivationFunc()
        {
            return "activ_res = 1 - (tanh(res) * tanh(res));";
        }
    }
}
