using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.ActivationFunctions
{
    [Serializable]
    public abstract class ActivationFunctionBase : IActivationFunction
    {
        private string func, func_deriv;


        public ActivationFunctionBase(string func, string func_deriv)
        {
            this.func = func;
            this.func_deriv = func_deriv;
        }

        public string Activation()
        {
            return func;
        }

        public string DerivActivation()
        {
            return func_deriv;
        }
    }
}
