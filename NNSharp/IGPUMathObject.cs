using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    public interface IGPUMathObject
    {
        CLEvent GetLatestEvent();
    }
}
