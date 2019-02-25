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
        public const float DefaultAlpha = 0.3f;

        public float Alpha { get; private set; }

        public LeakyReLU() : this(DefaultAlpha) { }
        public LeakyReLU(float alpha) : base() { Alpha = alpha; }

        protected override ActivationFunctionInfo ActivationFunc()
        {
            return new ActivationFunctionInfo()
            {
                GPUFunction = $"activ_res = isgreater(res, 0) * res + isless(res, 0) * {Alpha}f * res;",
                CPUFunction = (res) =>
                {
                    if (res < 0)
                        return res * Alpha;
                    else
                        return res;
                }
            };
        }

        protected override ActivationFunctionInfo DerivActivationFunc()
        {
            return new ActivationFunctionInfo()
            {
                GPUFunction = $"activ_res = isgreater(res, 0) + isless(res, 0) * {Alpha}f;",
                CPUFunction = (res) =>
                {
                    if (res < 0)
                        return Alpha;
                    else
                        return 1;
                }
            };
        }
    }
}
