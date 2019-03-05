using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Kernels
{
    public class KernelManager
    {
        //Design goals: Clean way to compile specific shaders for activation functions and loss functions
        static Dictionary<Tuple<int, int, int, int, int>, Kernel[]> conv_kernels;

        static Device device;

        public const int MaxWPT = 10;
        public const int Ratio = 128;

        public static bool GPUMode { get; set; }

        public static void Initialize()
        {
            conv_kernels = new Dictionary<Tuple<int, int, int, int, int>, Kernel[]>();
            device = Device.GetDevice();
        }

        #region Convolution
        const int MemLimit = 400;
        public static void Convolve(Matrix input, int input_off, int inputSz, Matrix kernel, int kernel_off, int kernel_side, bool rot180Kernel, int inputPadding, float stride, Matrix output, int output_off, int outputSize, bool rot180out, bool zero, Matrix bias, int bias_off)
        {
            Convolve(input.GPUMemory, input_off, inputSz, kernel.GPUMemory, kernel_off, kernel_side, rot180Kernel, inputPadding, stride, output.GPUMemory, output_off, outputSize, rot180out, zero, bias?.GPUMemory, bias_off);
        }

        private static void Convolve(Memory input, int input_off, int inputSz, Memory kernel, int kernelOff, int kernelSz, bool rot180Kernel, int inputPadding, float stride, Memory output, int output_off, int outputSize, bool rot180out, bool zero, Memory bias, int bias_off)
        {
            var a_dims = new Tuple<int, int, int, int, int>(inputSz, kernelSz, inputPadding, outputSize, (int)stride);
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
                    new string[] { "#define ADD_BIAS", "", "", "" },
                    new string[] { "#define ADD_BIAS", "", "", "#define ROT_KERN" },
                    new string[] { "#define ADD_BIAS", "", "#define ROT_OUT", "" },
                    new string[] { "#define ADD_BIAS", "", "#define ROT_OUT", "#define ROT_KERN" },
                    new string[] { "#define ADD_BIAS", "#define ZERO_O", "", "" },
                    new string[] { "#define ADD_BIAS", "#define ZERO_O", "", "#define ROT_KERN" },
                    new string[] { "#define ADD_BIAS", "#define ZERO_O", "#define ROT_OUT", "" },
                    new string[] { "#define ADD_BIAS", "#define ZERO_O", "#define ROT_OUT", "#define ROT_KERN" },
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
                idx |= 4;

            if (bias != null)
                idx |= (1 << 5);

            conv_kernels[a_dims][idx]
                .SetArgument(input_off)
                .SetArgument(kernelOff)
                .SetArgument(output_off)
                .SetArgument(bias_off)
                .SetArgumentMemory(input)
                .SetArgumentMemory(kernel)
                .SetArgumentMemory(output)
                .SetArgumentMemory(bias);

            device.Dispatch(conv_kernels[a_dims][idx], new uint[] { (uint)outputSize, (uint)outputSize }, null);
        }
        #endregion
    }
}
