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
        public LeakyReLU() : base("leaky_relu", "leaky_relu_deriv")
        {

        }
    }
}
