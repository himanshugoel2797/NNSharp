﻿using OpenCL.Net;
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
                if (!string.IsNullOrEmpty(defs[i]))
                    src += defs[i] + "\n";

            return CreateKernel(src + File.ReadAllText($"ANN/Kernels/CL/{file}.cl"), file, replace_val, out string err);
        }

        public Kernel CreateKernel(string code, string kernelName, string subs_name, out string err)
        {
            code = code.Replace("REPLACE_THIS", subs_name);

            //TODO: consider changing this so the kernel is only compiled on use, thus reducing the initialization time spent on kernel variations that aren't used in the current program
            Kernel kernel = new Kernel
            {
                Name = kernelName,

#if DELAY_COMPILE
                SourceCode = code,
                Initialized = false,
#else
                kern = env.Context.CompileKernelFromSource(code, kernelName, out err, out string errCode, "-cl-unsafe-math-optimizations"),
#endif
            };

#if DELAY_COMPILE
            string errCode = "Success";
            err = "";
#endif

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
            Memory m = new Memory
            {
                buf = Cl.CreateBuffer(env.Context, (MemFlags)(int)flags, (IntPtr)(len * sizeof(float)), out var errCode)
            };
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

#if DEBUG
        FileStream perfLogFile = null;
        StreamWriter perfLogWriter = null;
#endif
        public void Dispatch(Kernel k, uint[] global_sz, uint[] local_sz)
        {
#if DEBUG
            if (perfLogFile == null)
            {
                perfLogFile = File.OpenWrite("log.txt");
                perfLogWriter = new StreamWriter(perfLogFile);
            }
            System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
#endif

            Cl.EnqueueNDRangeKernel(env.CommandQueues[0], k.kern, 2, null, new IntPtr[] { (IntPtr)global_sz[0], (IntPtr)global_sz[1] }, local_sz == null ? null : new IntPtr[] { (IntPtr)local_sz[0], (IntPtr)local_sz[1] }, 0, null, out var eve);
            Cl.WaitForEvents((uint)1, new Event[] { eve });
            eve.Release();
            k.Reset();

#if DEBUG
            stopwatch.Stop();
            perfLogWriter.WriteLine($"Kernel Name: {k.Name}   Execution Time: {stopwatch.ElapsedTicks / (double)System.Diagnostics.Stopwatch.Frequency * 1000}ms");
            perfLogWriter.Flush();
#endif
        }
#endif
    }
}
