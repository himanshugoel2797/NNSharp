using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    class Program
    {
        static void Main(string[] args)
        {
            NNSharp.ANN.Kernels.KernelManager.Initialize();
            var superResolution = new FCSuperResolution();
            superResolution.Train();
        }
    }
}
