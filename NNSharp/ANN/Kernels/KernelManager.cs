using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Kernels
{
    public class KernelManager
    {
#if GPU
        public enum SGemvOperation
        {
            None,
            Add,
        }

        //Design goals: Clean way to compile specific shaders for activation functions and loss functions
        static Dictionary<Tuple<int, int>, Kernel[]> sgemv_kernels;
        static Dictionary<int, Kernel[]> vec_sum_kernels;
        static Dictionary<int, Kernel[]> vec_const_sum_kernels;
        static Dictionary<Tuple<int, int, int, int, int>, Kernel[]> conv_kernels;
        static Dictionary<string, Kernel[]> loss_kernels;
        static Dictionary<string, Kernel[]> activ_kernels;
        static Dictionary<string, Kernel[]> activ_hadamard_kernels;
        static Kernel[] fmop_kernels;
        static Kernel inner_prod;

        static Device device;

        public const int MaxWPT = 10;
        public const int Ratio = 128;

        public static void Initialize()
        {
            loss_kernels = new Dictionary<string, Kernel[]>();
            activ_kernels = new Dictionary<string, Kernel[]>();
            activ_hadamard_kernels = new Dictionary<string, Kernel[]>();
            sgemv_kernels = new Dictionary<Tuple<int, int>, Kernel[]>();
            conv_kernels = new Dictionary<Tuple<int, int, int, int, int>, Kernel[]>();
            vec_sum_kernels = new Dictionary<int, Kernel[]>();
            vec_const_sum_kernels = new Dictionary<int, Kernel[]>();

            device = Device.GetDevice();

            //FMOP
            //OPTIONS:
            //WPT values
            fmop_kernels = new Kernel[MaxWPT];
            for (int i = 0; i < MaxWPT; i++)
            {
                fmop_kernels[i] = device.LoadKernel("fmop", "", $"#define WPT ({1 << i})");
            }

            //Inner Product
            inner_prod = device.LoadKernel("inner_prod", "");
        }

        #region SGEMV
        //Generate specific kernels optimized for the dimension of the matrices in question
        //Matrix-Vector multiplication is memory bound
        //Have a single function that takes options for transposing, additional operations and activation functions
        public static void SGemv(Matrix a, Vector b, bool a_trans, Vector opt_c, SGemvOperation op, Vector output)
        {
            Tuple<int, int> a_dims = new Tuple<int, int>(a.Width, a.Height);
            if (!sgemv_kernels.ContainsKey(a_dims))
            {
                var l_sgemv_kernels = new Kernel[]{
                    device.LoadKernel("gemv", "", "#define S_OP_ADD", $"#define COLS ({a_dims.Item1})" , $"#define ROWS ({a_dims.Item2})"),
                    device.LoadKernel("gemv", "", "#define TRANSPOSE", $"#define COLS ({a_dims.Item1})" , $"#define ROWS ({a_dims.Item2})")
                };
                sgemv_kernels[a_dims] = l_sgemv_kernels;
            }

            //SGEMV
            //OPTIONS:
            //(a dot b) + c
            //(a^T dot b)
            if (a_trans && op == SGemvOperation.None)
            {
                sgemv_kernels[a_dims][1]
                    .SetArgumentMemory(a.memory)
                    .SetArgumentMemory(b.memory)
                    .SetArgumentMemory(b.memory)
                    .SetArgumentMemory(output.memory);

                device.Dispatch(sgemv_kernels[a_dims][1], new uint[] { (uint)a.Width, 1 }, null);
            }
            else if (!a_trans && op == SGemvOperation.Add)
            {
                sgemv_kernels[a_dims][0]
                    .SetArgumentMemory(a.memory)
                    .SetArgumentMemory(b.memory)
                    .SetArgumentMemory(opt_c.memory)
                    .SetArgumentMemory(output.memory);

                device.Dispatch(sgemv_kernels[a_dims][0], new uint[] { (uint)a.Height, 1 }, null);
            }
            else
                throw new Exception();
        }
        #endregion

        #region Vector Const Sum
        public static void VectorConstSum(Vector o, Vector i, int i_off)
        {
            var len = (MaxWPT - 1);
            while ((1 << len > o.Length | o.Length / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            if (!vec_const_sum_kernels.ContainsKey(o.Length))
            {
                var c_kerns = new Kernel[1];
                c_kerns[0] = device.LoadKernel("vector_const_sum", "", $"#define WPT ({1 << len})", $"#define O_LEN ({o.Length})");
                vec_const_sum_kernels[o.Length] = c_kerns;
            }

            vec_const_sum_kernels[o.Length][0]
                .SetArgumentMemory(o.memory)
                .SetArgumentMemory(i.memory)
                .SetArgument(i_off);

            device.Dispatch(vec_const_sum_kernels[o.Length][0], new uint[] { (uint)(o.Length / (1 << len) + 1), 1}, null);
        }
        #endregion

        #region Vector Sum
        public static void VectorSum(Vector o, int o_off, Vector i, int i_off, int i_len)
        {
            var len = (MaxWPT - 1);
            while ((1 << len > i_len | i_len / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            if (!vec_sum_kernels.ContainsKey(i_len))
            {
                var c_kerns = new Kernel[1];
                c_kerns[0] = device.LoadKernel("vector_sum", "", $"#define WPT ({1 << len})", $"#define I_LEN ({i_len})");
                vec_sum_kernels[i_len] = c_kerns;
            }

            vec_sum_kernels[i_len][0]
                .SetArgumentMemory(o.memory)
                .SetArgumentMemory(i.memory)
                .SetArgument(i_off)
                .SetArgument(o_off);

            device.Dispatch(vec_sum_kernels[i_len][0], new uint[] { (uint)(i_len / (1 << len) + 1), 1 }, null);
        }
        #endregion

        #region FMOP
        private static void Fmop(Memory a, float rate_a, Memory b, float rate_b, int a_len)
        {
            var len = (MaxWPT - 1);
            while ((1 << len > a_len | a_len / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            fmop_kernels[len]
                .SetArgument(a_len)
                .SetArgument(rate_a)
                .SetArgument(rate_b)
                .SetArgumentMemory(a)
                .SetArgumentMemory(b);

            device.Dispatch(fmop_kernels[len], new uint[] { (uint)(a_len / (1 << len)) + 1, 1 }, null);
        }

        public static void Fmop(Vector a, float rate_a, Vector b, float rate_b)
        {
            Fmop(a.memory, rate_a, b.memory, rate_b, a.Length);
        }

        public static void Fmop(Matrix a, float rate_a, Matrix b, float rate_b)
        {
            Fmop(a.memory, rate_a, b.memory, rate_b, a.Width * a.Height);
        }
        #endregion

        public static void Loss(Vector expectedOutput, Vector output, Vector loss, string func)
        {
            if (!loss_kernels.ContainsKey(func))
            {
                loss_kernels[func] = new Kernel[MaxWPT];
                for (int i = 0; i < MaxWPT; i++)
                    loss_kernels[func][i] = device.LoadKernel("loss", func, $"#define WPT ({1 << i})");
            }

            var len = (MaxWPT - 1);
            while ((1 << len > output.Length | output.Length / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            loss_kernels[func][len]
                .SetArgument(output.Length)
                .SetArgumentMemory(output.memory)
                .SetArgumentMemory(expectedOutput.memory)
                .SetArgumentMemory(loss.memory);

            device.Dispatch(loss_kernels[func][len], new uint[] { (uint)(expectedOutput.Length / (1 << len) + 1), 1 }, null);
        }

        public static void InnerProduct(Vector a, Vector b, Matrix c)
        {
            inner_prod
                .SetArgument(c.Width)
                .SetArgument(c.Height)
                .SetArgumentMemory(a.memory)
                .SetArgumentMemory(b.memory)
                .SetArgumentMemory(c.memory);

            device.Dispatch(inner_prod, new uint[] { (uint)c.Height, (uint)c.Width }, null);
        }

        #region Convolution
        const int MemLimit = 400;
        public static void Convolve(Vector input, int input_off, int inputSz, Matrix kernel, int kernel_off, int kernel_side, bool rot180Kernel, int inputPadding, int stride, Vector output, int output_off, int outputSize, bool rot180out, bool zero)
        {
            Convolve(input.memory, input_off, inputSz, kernel.memory, kernel_off, kernel_side, rot180Kernel, inputPadding, stride, output.memory, output_off, outputSize, rot180out, zero);
        }

        public static void Convolve(Vector input, int input_off, int inputSz, Vector kernel, int kernel_off, int kernel_side, bool rot180Kernel, int inputPadding, int stride, Matrix output, int output_off, int outputSize, bool rot180out, bool zero)
        {
            Convolve(input.memory, input_off, inputSz, kernel.memory, kernel_off, kernel_side, rot180Kernel, inputPadding, stride, output.memory, output_off, outputSize, rot180out, zero);
        }

        private static void Convolve(Memory input, int input_off, int inputSz, Memory kernel, int kernelOff, int kernelSz, bool rot180Kernel, int inputPadding, int stride, Memory output, int output_off, int outputSize, bool rot180out, bool zero)
        {
            var a_dims = new Tuple<int, int, int, int, int>(inputSz, kernelSz, inputPadding, outputSize, stride);
            if (!conv_kernels.ContainsKey(a_dims))
            {
                var common_args = new string[]
                {
                    $"#define IN_D ({a_dims.Item1})" , $"#define KERN_D ({a_dims.Item2})", $"#define IN_P ({a_dims.Item3})", $"#define OUT_D ({a_dims.Item4})", $"#define STRIDE ({a_dims.Item5})"
                };

                var diff_args = new string[][]
                {
                    new string[] { "", "", "" },
                    new string[] { "", "", "#define ROT_KERN" },
                    new string[] { "", "#define ROT_OUT", "" },
                    new string[] { "", "#define ROT_OUT", "#define ROT_KERN" },
                    new string[] { "#define ZERO_O", "", "" },
                    new string[] { "#define ZERO_O", "", "#define ROT_KERN" },
                    new string[] { "#define ZERO_O", "#define ROT_OUT", "" },
                    new string[] { "#define ZERO_O", "#define ROT_OUT", "#define ROT_KERN" },
                };

                var conv_kern_names = new string[] { "conv", "conv_ksmall", "conv_ismall" };

                var l_conv_kernels = new List<Kernel>();
                for (int i = 0; i < conv_kern_names.Length; i++)
                    for (int j = 0; j < diff_args.Length; j++)
                    {
                        var kern_args = new List<string>();
                        kern_args.AddRange(diff_args[j]);
                        kern_args.AddRange(common_args);

                        l_conv_kernels.Add(device.LoadKernel(conv_kern_names[i], "", kern_args.ToArray()));
                    }
                conv_kernels[a_dims] = l_conv_kernels.ToArray();
            }

            int idx = 0;
            if (kernelSz * kernelSz < MemLimit)
                idx = 1 << 3;
            else if (inputSz * inputSz < MemLimit)
                idx = 2 << 3;
            else
                idx = 0;

            if (rot180Kernel)
                idx |= 1;

            if (rot180out)
                idx |= 2;

            if (zero)
                idx |= 7;


            conv_kernels[a_dims][idx]
                .SetArgument(input_off)
                .SetArgument(kernelOff)
                .SetArgument(output_off)
                .SetArgumentMemory(input)
                .SetArgumentMemory(kernel)
                .SetArgumentMemory(output);

            device.Dispatch(conv_kernels[a_dims][idx], new uint[] { (uint)outputSize, (uint)outputSize }, null);
        }
        #endregion

        #region Activations
        public static void Activ(Vector input, Vector output, ActivationFunctionInfo func_info)
        {
            string func = func_info.GPUFunction;

            if (!activ_kernels.ContainsKey(func))
            {
                activ_kernels[func] = new Kernel[MaxWPT];
                for (int i = 0; i < MaxWPT; i++)
                    activ_kernels[func][i] = device.LoadKernel("activ", func, $"#define WPT ({1 << i})");
            }

            var len = (MaxWPT - 1);
            while ((1 << len > output.Length | output.Length / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            activ_kernels[func][len]
                .SetArgument(output.Length)
                .SetArgumentMemory(input.memory)
                .SetArgumentMemory(output.memory);

            device.Dispatch(activ_kernels[func][len], new uint[] { (uint)(output.Length / (1 << len) + 1), 1 }, null);
        }

        public static void HadamardActiv(Vector a, Vector input, Vector output, ActivationFunctionInfo func_info)
        {
            string func = func_info.GPUFunction;

            if (!activ_hadamard_kernels.ContainsKey(func))
            {
                activ_hadamard_kernels[func] = new Kernel[MaxWPT];
                for (int i = 0; i < MaxWPT; i++)
                    activ_hadamard_kernels[func][i] = device.LoadKernel("activ", func, $"#define WPT ({1 << i})", "#define HADAMARD");
            }

            var len = (MaxWPT - 1);
            while ((1 << len > output.Length | output.Length / (1 << len) < Ratio) && len >= 0)
                len--;
            if (len < 0) len = 0;

            activ_hadamard_kernels[func][len]
                .SetArgument(output.Length)
                .SetArgumentMemory(input.memory)
                .SetArgumentMemory(a.memory)
                .SetArgumentMemory(output.memory);

            device.Dispatch(activ_hadamard_kernels[func][len], new uint[] { (uint)(output.Length / (1 << len) + 1), 1 }, null);
        }
        #endregion
#elif CPU
        public static void Initialize() { }
#endif
    }
}
