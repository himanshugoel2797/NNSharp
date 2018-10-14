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
        public Sigmoid() : base("sigmoid", "sigmoid_deriv") { }
    }
}
