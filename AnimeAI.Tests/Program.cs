using NNSharp.ANN.Kernels;
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
            KernelManager.Initialize();
            KernelManager.GPUMode = true;
            //var superResolution = new GAN();//new ReversibleAutoencoder(); //new ConvSuperResolution();
            new ConvSuperResolution().Train();
            //new GradientChecking().Check();
        }
    }
}
