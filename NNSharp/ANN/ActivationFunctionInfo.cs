using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public class ActivationFunctionInfo
    {
        public Func<float, float> CPUFunction { get; set; }
        public string GPUFunction { get; set; }
    }
}
