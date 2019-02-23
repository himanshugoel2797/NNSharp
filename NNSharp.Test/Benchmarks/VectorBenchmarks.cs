using NNSharp.ANN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Benchmarks
{
    class VectorBenchmarks
    {
        public void RunAll(int len, int runCnt, IActivationFunction func)
        {
            var add = new BenchmarkHarness("Add");
            var msub = new BenchmarkHarness("MSub");
            var activ = new BenchmarkHarness("Activation");
            var deriv_activ = new BenchmarkHarness("DerivActivation");

            for (int i = 0; i < runCnt; i++)
            {
                Add(len, add);
                MSub(len, msub);
                Activation(len, func, activ);
                DerivActivation(len, func, deriv_activ);
            }
            
            add.Show();
            msub.Show();
            activ.Show();
            deriv_activ.Show();
        }

        public void Add(int len, BenchmarkHarness harness)
        {
            Vector a = new Vector(len, MemoryFlags.ReadWrite, true);
            Vector b = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Vector.Add(a, b);
            harness.Stop();
        }

        public void MSub(int len, BenchmarkHarness harness)
        {
            Vector a = new Vector(len, MemoryFlags.ReadWrite, true);
            Vector b = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Vector.MSubSelf(a, b, 0);
            harness.Stop();
        }

        public void Activation(int len, IActivationFunction func, BenchmarkHarness harness)
        {
            Vector a = new Vector(len, MemoryFlags.ReadWrite, true);
            Vector b = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Vector.Activation(a, b, func.Activation());
            harness.Stop();
        }

        public void DerivActivation(int len, IActivationFunction func, BenchmarkHarness harness)
        {
            Vector a = new Vector(len, MemoryFlags.ReadWrite, true);
            Vector b = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Vector.Activation(a, b, func.DerivActivation());
            harness.Stop();
        }
    }
}
