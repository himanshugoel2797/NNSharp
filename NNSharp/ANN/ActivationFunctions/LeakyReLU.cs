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
        public LeakyReLU() : base("const float activ_res = isgreater(res, 0) * res + isless(res, 0) * 0.01f * res;", "const float activ_res = isgreater(res, 0) + isless(res, 0) * 0.01f;")
        {

        }
    }
}
