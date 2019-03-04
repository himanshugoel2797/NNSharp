using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NNSharp.Test
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            ANN.Kernels.KernelManager.Initialize();
            new MNIST.DGAN().Train();
            Console.ReadLine();
        }
    }
}
