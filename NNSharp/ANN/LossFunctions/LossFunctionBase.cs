using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    [Serializable]
    public abstract class LossFunctionBase : ILossFunction
    {
        private string func, func_deriv;

        public LossFunctionBase(string func, string func_deriv)
        {
            this.func = func;
            this.func_deriv = func_deriv;
        }

        public void Loss(Vector output, Vector expectedOutput, Vector result)
        {
            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(output.Length);
            var optTS = Device.OptimalTS(func, output.Length, false, true);
            dev[func, optWPT, optTS].SetArgumentMemory(output.memory)
                                 .SetArgumentMemory(expectedOutput.memory)
                                 .SetArgumentMemory(result.memory);

            dev.Dispatch(dev[func, optWPT, optTS], new uint[] { (uint)output.Length / optWPT, 1 }, new uint[] { optTS, 1 });
        }

        public void LossDeriv(Vector output, Vector expectedOutput, Vector result)
        {
            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(output.Length);
            var optTS = Device.OptimalTS(func_deriv, output.Length, false, true);
            dev[func_deriv, optWPT, optTS].SetArgumentMemory(output.memory)
                                 .SetArgumentMemory(expectedOutput.memory)
                                 .SetArgumentMemory(result.memory);

            dev.Dispatch(dev[func_deriv, optWPT, optTS], new uint[] { (uint)output.Length / optWPT, 1 }, new uint[] { optTS, 1 });
        }
    }
}
