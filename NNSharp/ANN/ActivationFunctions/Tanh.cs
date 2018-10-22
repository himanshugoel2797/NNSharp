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
        public Tanh() : base("const float activ_res = tanh(res);", "const float activ_res = 1 - (tanh(res) * tanh(res));") { }
    }
}
