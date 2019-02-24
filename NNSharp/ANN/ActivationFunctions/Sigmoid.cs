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
        public Sigmoid() : base() { }

        protected override ActivationFunctionInfo ActivationFunc()
        {
            return new ActivationFunctionInfo()
            {
                GPUFunction = "activ_res = 1.0f / (1.0f + exp(-res));",
                CPUFunction = (res) =>
                {
                    return (float)(1.0d / (1.0d + Math.Exp(-res)));
                }
            };
        }

        protected override ActivationFunctionInfo DerivActivationFunc()
        {
            return new ActivationFunctionInfo()
            {
                GPUFunction = "const float res_a = 1.0f / (1.0f + exp(-res)); activ_res = res_a * (1.0f - res_a);",
                CPUFunction = (res) =>
                {
                    var tmp = (1.0d / (1.0d + Math.Exp(-res)));
                    return (float)(tmp * (1 - tmp));
                }
            };
        }
    }
}
