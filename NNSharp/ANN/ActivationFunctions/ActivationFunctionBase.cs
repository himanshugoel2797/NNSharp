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

        public void Activation(Vector o)
        {
            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(o.Length);
            var optTS = Device.OptimalTS(func, o.Length, false, true);
            dev[func, optWPT, optTS].SetArgumentMemory(o.memory);

            dev.Dispatch(dev[func, optWPT, optTS], new uint[] { (uint)(o.Length / optWPT), 1 }, new uint[] { optTS, 1 });
        }

        public void DerivActivation(Vector o)
        {
            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(o.Length);
            var optTS = Device.OptimalTS(func_deriv, o.Length, false, true);
            dev[func_deriv, optWPT, optTS].SetArgumentMemory(o.memory);

            dev.Dispatch(dev[func_deriv, optWPT, optTS], new uint[] { (uint)(o.Length / optWPT), 1 }, new uint[] { optTS, 1 });
        }
    }
}
