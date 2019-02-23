using NNSharp.ANN.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Benchmarks
{
    class BenchmarkRunner : ITest
    {
        public void Run()
        {
            using (StreamWriter writer = new StreamWriter(DateTime.Now.Ticks + ".txt"))
            {
                Console.SetOut(writer);

                VectorBenchmarks vectorBenchmarks = new VectorBenchmarks();
                MatrixBenchmarks matrixBenchmarks = new MatrixBenchmarks();

                Console.WriteLine("Vector ");
                for (int i = 8; i < 18; i++)
                {
                    Console.WriteLine($"N = {1 << i}");
                    vectorBenchmarks.RunAll(1 << i, 1000, new Sigmoid());
                }

                Console.WriteLine("Matrix ");
                for (int i = 8; i < 12; i++)
                {
                    Console.WriteLine($"N = {1 << i}");
                    matrixBenchmarks.RunAll(1 << i, 1000);
                }
            }
        }
    }
}
