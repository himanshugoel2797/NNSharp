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
        public ReLU() : base("const float activ_res = isgreater(res, 0) * res;", "const float activ_res = isgreater(res, 0);")
        {

        }
    }
}
