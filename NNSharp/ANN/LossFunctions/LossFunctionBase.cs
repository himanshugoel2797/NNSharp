using NNSharp.ANN.Kernels;
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
#if GPU
            KernelManager.Loss(expectedOutput, output, result, func);
#elif CPU
            switch (func)
            {
                case "quadratic_loss":
                    {
                        Parallel.For(0, result.Length, (i) => result.memory[i] += (0.5f * (float)Math.Pow(output.memory[i] - expectedOutput.memory[i], 2))/result.Length);
                    }
                    break;
                case "binary_cross_entropy":
                    {
                        //- (z * log(y + eps) + (1-z) * log(1 - y + eps))
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Length, (i) =>
                        {
                            result.memory[i] += (-(float)(expectedOutput.memory[i] * Math.Log(output.memory[i] + float.Epsilon) + (1 - expectedOutput.memory[i]) * Math.Log(1 - output.memory[i] + float.Epsilon)) / result.Length);
                        });
                    }
                    break;
            }
#endif
        }

        public void LossDeriv(Vector output, Vector expectedOutput, Vector result)
        {
#if GPU
            KernelManager.Loss(expectedOutput, output, result, func_deriv);
#elif CPU
            switch (func_deriv)
            {
                case "quadratic_loss_deriv":
                    {
                        Parallel.For(0, result.Length, (i) => result.memory[i] += (output.memory[i] - expectedOutput.memory[i]) / result.Length);
                    }
                    break;
                case "binary_cross_entropy_deriv":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Length, (i) =>
                        {
                            float r_0 = ((1.0f - expectedOutput.memory[i]) - output.memory[i]);
                            float r = (r_0 == 0) ? 0 : (1.0f / r_0);
                            result.memory[i] += r;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Length);
                        });
                    }
                    break;
            }
#endif
        }
    }
}
