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
        private readonly string func;
        private readonly string func_deriv;

        public LossFunctionBase(string func, string func_deriv)
        {
            this.func = func;
            this.func_deriv = func_deriv;
        }

        public void Loss(Matrix output, Matrix expectedOutput, Matrix result, float regularizationParam)
        {
#if GPU
            KernelManager.Loss(expectedOutput, output, result, func);
#elif CPU
            switch (func)
            {
                case "quadratic_loss":
                    {
                        Parallel.For(0, result.Rows, (i) => result.Memory[i] += (0.5f * (float)Math.Pow(output.Memory[i] - expectedOutput.Memory[i], 2))/result.Rows + regularizationParam);
                    }
                    break;
                case "binary_cross_entropy":
                    {
                        //- (z * log(y + eps) + (1-z) * log(1 - y + eps))
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            result.Memory[i] += (-(float)(expectedOutput.Memory[i] * Math.Log(output.Memory[i] + float.Epsilon) + (1 - expectedOutput.Memory[i]) * Math.Log(1 - output.Memory[i] + float.Epsilon)) / result.Rows + regularizationParam);
                        });
                    }
                    break;
                case "gan_disc_fake":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = (float)Math.Log(1.0f - expectedOutput.Memory[i] - output.Memory[i]);
                            result.Memory[i] += r_0 + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
                case "gan_disc_real":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = -(float)Math.Log((1 - expectedOutput.Memory[i]) + output.Memory[i]);
                            result.Memory[i] += r_0 + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
                case "gan_gen":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = -(float)Math.Log(output.Memory[i]);
                            result.Memory[i] += r_0 + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
            }
#endif
        }

        public void LossDeriv(Matrix output, Matrix expectedOutput, Matrix result, float regularizationParam)
        {
#if GPU
            KernelManager.Loss(expectedOutput, output, result, func_deriv);
#elif CPU
            switch (func_deriv)
            {
                case "quadratic_loss_deriv":
                    {
                        Parallel.For(0, result.Rows, (i) => result.Memory[i] += (output.Memory[i] - expectedOutput.Memory[i]) / result.Rows + regularizationParam);
                    }
                    break;
                case "binary_cross_entropy_deriv":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = ((1.0f - expectedOutput.Memory[i]) - output.Memory[i]);
                            float r = (r_0 == 0) ? 0 : (1.0f / r_0);
                            result.Memory[i] += r + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
                case "gan_disc_fake":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = (1.0f - expectedOutput.Memory[i] - output.Memory[i]);
                            float r = (r_0 == 0) ? 0 : (1.0f / r_0);
                            result.Memory[i] += r + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
                case "gan_disc_real":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = ((1 - expectedOutput.Memory[i]) + output.Memory[i]);
                            float r = (r_0 == 0) ? 0 : -(1.0f / r_0);
                            result.Memory[i] += r + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
                case "gan_gen":
                    {
                        //- (z / (y + eps) - (1 - z) / (1 - y + eps));
                        //z = expected out
                        //y = actual output
                        Parallel.For(0, result.Rows, (i) =>
                        {
                            float r_0 = -1.0f/(output.Memory[i]);
                            result.Memory[i] += r_0 + regularizationParam;
                            //result.memory[i] += -(float)((expectedOutput.memory[i] / (output.memory[i] + float.Epsilon) - (1 - expectedOutput.memory[i]) / (1 - output.memory[i] + float.Epsilon))/result.Rows);
                        });
                    }
                    break;
            }
#endif
        }
    }
}
