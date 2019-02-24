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
        public ActivationFunctionBase()
        {
        }

        protected abstract ActivationFunctionInfo ActivationFunc();
        protected abstract ActivationFunctionInfo DerivActivationFunc();

        public ActivationFunctionInfo Activation()
        {
            return ActivationFunc();
        }

        public ActivationFunctionInfo DerivActivation()
        {
            return DerivActivationFunc();
        }
    }
}
