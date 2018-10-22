using OpenCL.Net;
using OpenCL.Net.Extensions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    public enum MemoryFlags
    {
        ReadWrite = 1,
        WriteOnly = 2,
        ReadOnly = 4,
    }

    public class Device
    {
        private OpenCL.Net.Environment env;
        private Dictionary<string, Kernel> kernels;

        private Device() { kernels = new Dictionary<string, Kernel>(); }

        public const int WPT = 8;
        public const int TS = 256;

        public const int MatrixMultTS = 5;
        public const int MatrixMultWPT = 1;

        private static Device device;
        public static Device GetDevice()
        {
            if (device == null)
            {
                device = new Device()
                {
                    env = "*".CreateCLEnvironment(OpenCL.Net.DeviceType.Gpu)
                };
                device.LoadAllKernels();
            }

            return device;
        }

        public static uint OptimalWPT(int len)
        {
            if (len % 8 == 0)
                return 8;
            else if (len % 4 == 0)
                return 4;
            else if (len % 2 == 0)
                return 2;
            else
                return 1;
        }

        private static bool kernExists(string kern, int ts, uint wpt)
        {
            return device.kernels.ContainsKey(kern + "_" + wpt + "_" + ts);
        }

        public static uint OptimalTS(string kern, int len, bool root, bool wpt)
        {
            uint wpt_val = 1;
            if (wpt)
            {
                wpt_val = OptimalWPT(len);
                len /= (int)wpt_val;
            }

            if (!root)
            {
                if (len % 256 == 0 && kernExists(kern, 256, wpt_val))
                    return 256;
                if (len % 128 == 0 && kernExists(kern, 128, wpt_val))
                    return 128;
                if (len % 64 == 0 && kernExists(kern, 64, wpt_val))
                    return 64;
                if (len % 32 == 0 && kernExists(kern, 32, wpt_val))
                    return 32;
            }
            if (len % 16 == 0 && kernExists(kern, 16, wpt_val))
                return 16;
            if (len % 8 == 0 && kernExists(kern, 8, wpt_val))
                return 8;
            else if (len % 4 == 0 && kernExists(kern, 4, wpt_val))
                return 4;
            else if (len % 2 == 0 && kernExists(kern, 2, wpt_val))
                return 2;
            else
                return 1;
        }

        public Kernel this[string name, uint wpt, uint ts]
        {
            get { return kernels[name + "_" + wpt + "_" + ts]; }
        }

        public string LoadKernel(string file, string substr, int ts_limit = 9, int wpt_limit = 4)
        {
            for (int i = 0; i < wpt_limit; i++)
            {
                for (int ts_n = 0; ts_n < ts_limit; ts_n++)
                {
                    CreateKernel(File.ReadAllText($"Kernels/{file}.cl").Replace("REPLACE_THIS", substr), file, substr, 1u << i, 1u << ts_n, out string err);

                    if (!string.IsNullOrEmpty(err))
                        return err;
                }
            }
            return "";
        }

        public string LoadKernel(string file, int ts_limit = 9, int wpt_limit = 4)
        {
            for (int i = 0; i < wpt_limit; i++)
            {
                for (int ts_n = 0; ts_n < ts_limit; ts_n++)
                {
                    CreateKernel(File.ReadAllText($"Kernels/{file}.cl"), file, "", 1u << i, 1u << ts_n, out string err);

                    if (!string.IsNullOrEmpty(err))
                        return err;
                }
            }
            return "";
        }

        public void LoadAllKernels()
        {
            LoadKernel("gmm");
            LoadKernel("mv_madd", MatrixMultTS, MatrixMultWPT);
            LoadKernel("tmv_mmult", MatrixMultTS, MatrixMultWPT);
            LoadKernel("v_hadamard");
            LoadKernel("v_msub");
            LoadKernel("v_div");
            LoadKernel("v_add");
            LoadKernel("m_msub");
            LoadKernel("vv_mmult");

            LoadKernel("quadratic_loss");
            LoadKernel("quadratic_loss_deriv");

            LoadKernel("gan_disc_crossentropy_loss_deriv");

            /*LoadKernel("relu");
            LoadKernel("relu_deriv");

            LoadKernel("leaky_relu");
            LoadKernel("leaky_relu_deriv");

            LoadKernel("sigmoid");
            LoadKernel("sigmoid_deriv");

            LoadKernel("tanh_act");
            LoadKernel("tanh_deriv");*/
        }

        public Kernel CreateKernel(string code, string kernelName, string subs_name, uint wpt, uint ts, out string err)
        {
            string hash_str = string.IsNullOrWhiteSpace(subs_name) ? "" : "_" + Math.Abs(subs_name.GetHashCode());
            if (kernels.ContainsKey(kernelName + hash_str + "_" + wpt + "_" + ts))
            {
                err = null;
                return kernels[kernelName + hash_str + "_" + wpt + "_" + ts];
            }

            code = $@"#define WPT {wpt}
#define TS {ts}
#define RTS ({ts / wpt})

" + code;
            string errCode;
            Kernel kernel = new Kernel
            {
                kern = env.Context.CompileKernelFromSource(code, kernelName, out err, out errCode, "-cl-unsafe-math-optimizations"),
                wpt = wpt
            };

            if (errCode == "Success")
            {
                kernels[kernelName + hash_str + "_" + wpt + "_" + ts] = kernel;
                if (!string.IsNullOrWhiteSpace(err))
                    throw new Exception(err);

                return kernel;
            }
            else
                return null;
        }

        public Memory AllocateMemory(int len, MemoryFlags flags, bool zero)
        {
            Memory m = new Memory();
            m.buf = Cl.CreateBuffer(env.Context, (MemFlags)(int)flags, (IntPtr)(len * sizeof(float)), out var errCode);
            if (errCode != ErrorCode.Success)
                throw new Exception(errCode.ToString());
            return m;
        }

        public void Write(Memory mem, float[] data, int offset = 0)
        {
            env.CommandQueues[1].WriteToBuffer(mem.buf, data, offset, -1, new Event[0]);
        }

        public void Read(Memory mem, float[] data)
        {
            env.CommandQueues[2].ReadFromBuffer(mem.buf, data, 0, -1, new Event[0]);
        }

        public void Dispatch(Kernel k, uint[] global_sz, uint[] local_sz)
        {
            Cl.EnqueueNDRangeKernel(env.CommandQueues[0], k.kern, 2, null, new IntPtr[] { (IntPtr)global_sz[0], (IntPtr)global_sz[1] }, new IntPtr[] { (IntPtr)local_sz[0], (IntPtr)local_sz[1] }, 0, null, out var eve);
            Cl.WaitForEvents((uint)1, new Event[] { eve });
            eve.Release();
        }
    }
}
