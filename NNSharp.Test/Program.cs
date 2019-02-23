using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Test.Autoencoders;
using NNSharp.Test.GANs;
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

            /*Matrix a = new Matrix(2, 2, MemoryFlags.ReadWrite, false);
            a.Write(new float[] { 1, 2, 3, 4 });

            Vector b = new Vector(2, MemoryFlags.ReadWrite, false);
            b.Write(new float[] { 5, 6 });

            Vector c = new Vector(2, MemoryFlags.ReadWrite, false);
            c.Write(new float[] { 7, 8 });

            Vector res = new Vector(2, MemoryFlags.ReadWrite, false);
            Vector res2 = new Vector(2, MemoryFlags.ReadWrite, false);
            Matrix res3 = new Matrix(2, 2, MemoryFlags.ReadWrite, false);

            Matrix.TMmult(a, b, res);
            Matrix.Madd(a, b, c, res2);
            Matrix.MatrixProduct(b, c, res3);*/

            //ITest t = new Classifiers.TextClassifier();
            //var t = new Benchmarks.DeepNetworkCheck();// ConvNetworkCheck();
            //var t = new MathTests.Convolution();
            //var t = new HeadlessDGAN();
            //t.Run();

            //var a = new MathTests.Convolution(); 
            //a.Run();
            var a = new OLD.ConvolutionalAutoencoder();
            a.Train();

            //Benchmarks.BenchmarkRunner benchmarkRunner = new Benchmarks.BenchmarkRunner();
            //benchmarkRunner.Run();

            //Benchmarks.DeepNetworkCheck check = new Benchmarks.DeepNetworkCheck();
            //check.Run();

            Console.ReadLine();
        }
    }
}
