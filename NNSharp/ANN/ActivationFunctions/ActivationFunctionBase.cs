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

        protected abstract string ActivationFunc();
        protected abstract string DerivActivationFunc();

        public string Activation()
        {
#if CPU
            return func;
#elif GPU
            return ActivationFunc();
#endif
        }

        public string DerivActivation()
        {
#if CPU
            return func_deriv;
#elif GPU
            return DerivActivationFunc();
#endif
        }
    }
}
