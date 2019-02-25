using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;
using OpenCL.Net.Extensions;

namespace NNSharp
{
    public class Kernel
    {
        public string Name { get; set; }

#if DELAY_COMPILE && GPU
        public string SourceCode { get; set; }
        public bool Initialized { get; set; }
        internal Event pendingExecution { get; set; }
#endif

        internal OpenCL.Net.Kernel kern;
        private CLExtensions.KernelArgChain chain;
        private bool reset = true;

        public void Reset()
        {
            reset = true;
        }

        public Kernel SetArgument<T>(T val) where T : struct, IComparable
        {
#if GPU
#if DELAY_COMPILE
            if (!Initialized)
            {
                Initialized = true;
                kern = Device.GetDevice().env.Context.CompileKernelFromSource(SourceCode, Name, "-cl-unsafe-math-optimizations");
            }
#endif
            if (pendingExecution.IsValid())
            {
                Device.GetDevice().HandleEvent();
            }
#endif

            if (reset)
            {
                chain = kern.SetKernelArg(val);
                reset = false;
            }
            else
                chain = chain.SetKernelArg(val);
            return this;
        }

        public Kernel SetArgumentMemory(Memory val)
        {
#if GPU
#if DELAY_COMPILE
            if (!Initialized)
            {
                Initialized = true;
                kern = Device.GetDevice().env.Context.CompileKernelFromSource(SourceCode, Name, "-cl-unsafe-math-optimizations");
            }
#endif
            if (pendingExecution.IsValid())
            {
                Device.GetDevice().HandleEvent();
            }
#endif
            if (reset)
            {
                chain = kern.SetKernelArg((IMem)val.buf);
                reset = false;
            }
            else
                chain = chain.SetKernelArg((IMem)val.buf);
            return this;
        }
    }
}
