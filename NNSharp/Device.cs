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
#if GPU
        internal OpenCL.Net.Environment env;

        private Device() { }

        private static Device device;
        public static Device GetDevice()
        {
            if (device == null)
            {
                device = new Device()
                {
                    env = "*".CreateCLEnvironment(OpenCL.Net.DeviceType.Gpu)
                };
            }

            return device;
        }

        public Kernel LoadKernel(string file, string replace_val = "", params string[] defs)
        {
            string src = "";
            for (int i = 0; i < defs.Length; i++)
                src += defs[i] + "\n";

            return CreateKernel(src + File.ReadAllText($"ANN/Kernels/CL/{file}.cl"), file, replace_val, out string err);
        }

        public Kernel CreateKernel(string code, string kernelName, string subs_name, out string err)
        {
            code = code.Replace("REPLACE_THIS", subs_name);

            string errCode;
            Kernel kernel = new Kernel
            {
                kern = env.Context.CompileKernelFromSource(code, kernelName, out err, out errCode, "-cl-unsafe-math-optimizations"),
            };

            if (errCode == "Success")
            {
                if (!string.IsNullOrWhiteSpace(err))
                    throw new Exception(err);

                return kernel;
            }
            else
            { 
                throw new Exception(err);
            }
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
            Cl.EnqueueNDRangeKernel(env.CommandQueues[0], k.kern, 2, null, new IntPtr[] { (IntPtr)global_sz[0], (IntPtr)global_sz[1] }, local_sz == null ? null : new IntPtr[] { (IntPtr)local_sz[0], (IntPtr)local_sz[1] }, 0, null, out var eve);
            Cl.WaitForEvents((uint)1, new Event[] { eve });
            eve.Release();
            k.Reset();
        }
#endif
    }
}
