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
        public Tanh() : base("tanh_act", "tanh_deriv") { }
    }
}
