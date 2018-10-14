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
        internal OpenCL.Net.Kernel kern;
        internal uint wpt;

        public KernelArgumentChain SetArgument<T>(T val) where T : struct, IComparable
        {
            return new KernelArgumentChain(kern.SetKernelArg(val));
        }

        public KernelArgumentChain SetArgumentMemory(Memory val)
        {
            return new KernelArgumentChain(kern.SetKernelArg((IMem)val.buf));
        }
    }

    public class KernelArgumentChain
    {
        private CLExtensions.KernelArgChain argChain;

        internal KernelArgumentChain(CLExtensions.KernelArgChain argChain)
        {
            this.argChain = argChain;
        }

        public KernelArgumentChain SetArgument<T>(T val) where T : struct, IComparable
        {
            argChain = argChain.SetKernelArg(val);
            return this;
        }

        public KernelArgumentChain SetArgumentMemory(Memory val)
        {
            argChain = argChain.SetKernelArg((IMem)val.buf);
            return this;
        }
    }
}
