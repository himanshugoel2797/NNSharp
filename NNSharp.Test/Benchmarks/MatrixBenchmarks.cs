using NNSharp.ANN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Benchmarks
{
    class MatrixBenchmarks
    {
        public void RunAll(int len, int runCnt)
        {
            var madd = new BenchmarkHarness("MAdd");
            var tmmult = new BenchmarkHarness("TMMult");
            var msub = new BenchmarkHarness("MSub");
            var matProd = new BenchmarkHarness("Matrix Product");
            
            for (int i = 0; i < runCnt; i++)
            {
                MAdd(len, madd);
                TMMult(len, tmmult);
                MSub(len, msub);
                MatProd(len, matProd);
            }

            madd.Show();
            tmmult.Show();
            msub.Show();
            matProd.Show();
        }

        public void MAdd(int len, BenchmarkHarness harness)
        {
            var a = new Matrix(len, len, MemoryFlags.ReadWrite, true);
            var b = new Vector(len, MemoryFlags.ReadWrite, true);
            var c = new Vector(len, MemoryFlags.ReadWrite, true);
            var d = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Matrix.Madd(a, b, c, d);
            harness.Stop();
        }

        public void TMMult(int len, BenchmarkHarness harness)
        {
            var a = new Matrix(len, len, MemoryFlags.ReadWrite, true);
            var b = new Vector(len, MemoryFlags.ReadWrite, true);
            var d = new Vector(len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Matrix.TMmult(a, b, d);
            harness.Stop();
        }

        public void MSub(int len, BenchmarkHarness harness)
        {
            var a = new Matrix(len, len, MemoryFlags.ReadWrite, true);
            var b = new Matrix(len, len, MemoryFlags.ReadWrite, true);

            harness.Start();
            Matrix.MSubSelf(a, b, 0);
            harness.Stop();
        }

        public void MatProd(int len, BenchmarkHarness harness)
        {
            Vector a = new Vector(len, MemoryFlags.ReadWrite, true);
            Vector b = new Vector(len, MemoryFlags.ReadWrite, true);
            Matrix c = new Matrix(len, len, MemoryFlags.ReadWrite, false);

            harness.Start();
            Matrix.MatrixProduct(a, b, c);
            harness.Stop();
        }
    }
}
