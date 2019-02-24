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
        public Tanh() : base() { }

        protected override ActivationFunctionInfo ActivationFunc()
        {
            return new ActivationFunctionInfo(){
                GPUFunction = "activ_res = tanh(res);",
                CPUFunction = (res) =>
                {
                    return (float)(Math.Tanh(res));
                }
            };
        }

        protected override ActivationFunctionInfo DerivActivationFunc()
        {
            return new ActivationFunctionInfo() {
                GPUFunction = "activ_res = 1 - (tanh(res) * tanh(res));",
                CPUFunction = (res) =>
                {
                    var tmp = (float)(Math.Tanh(res));
                    return (1 - tmp * tmp);
                }
            };
        }
    }
}
