using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NNSharp.Tests
{
    [TestClass]
    public class MatrixTests
    {
        const int M = 320;
        const int K = (16);
        const int N = 640;

        [TestMethod]
        public void MatrixVectorMadd()
        {
            Matrix a = new Matrix(N, M, MemoryFlags.ReadOnly, false);
            Vector b = new Vector(N, MemoryFlags.ReadOnly, false);
            Vector c = new Vector(M, MemoryFlags.ReadOnly, true);

            Vector d = new Vector(M, MemoryFlags.WriteOnly, false);
            Random rng = new Random(0);

            float[] a_data = new float[a.Width * a.Height];
            float[] b_data = new float[b.Length];
            float[] c_data = new float[c.Length];

            float[] d_data = new float[d.Length];
            float[] d_data_gpu = new float[d.Length];

            for (int i = 0; i < a_data.Length; i++)
                a_data[i] = (float)rng.NextDouble();

            for (int i = 0; i < b_data.Length; i++)
                b_data[i] = (float)rng.NextDouble();

            a.Write(a_data);
            b.Write(b_data);
            Matrix.Madd(a, b, c, d);
            d.Read(d_data_gpu);

            for (int i = 0; i < M; i++)
            {
                float dot = 0;
                 
                for (int j = 0; j < N; j++) 
                {
                    dot += (a_data[j * M + i] * b_data[j]);
                }

                d_data[i] = dot + c_data[i];
            } 
             
            var fstream = File.Open("log.txt", FileMode.Create);
            var fwriter = new StreamWriter(fstream);

            for (int n = 0; n < d_data.Length; n++)
                fwriter.Write("\t" + d_data[n] + "\t");

            fwriter.WriteLine();
            for (int n = 0; n < d_data_gpu.Length; n++)
                fwriter.Write("\t" + d_data_gpu[n] + "\t");

            fwriter.Flush();
            fwriter.Close();
             
            for (int w = 0; w < d_data.Length; w++)
                if (Math.Abs(d_data[w] - d_data_gpu[w])/((d_data[w] + d_data_gpu[w]) * 0.5f) > 0.005f)
                    Assert.Fail();
        }

        [TestMethod]
        public void TMatrixVectorMmult()
        {
            Matrix a = new Matrix(N, M, MemoryFlags.ReadOnly, false);
            Vector b = new Vector(M, MemoryFlags.ReadOnly, false);
            Vector c = new Vector(N, MemoryFlags.ReadOnly, false);

            Vector d = new Vector(N, MemoryFlags.WriteOnly, false);
            Random rng = new Random(0);


            float[] a_data = new float[a.Width * a.Height];
            float[] b_data = new float[b.Length];
            float[] c_data = new float[c.Length];

            float[] d_data = new float[d.Length];
            float[] d_data_gpu = new float[d.Length];

            for (int i = 0; i < a_data.Length; i++)
                a_data[i] = (float)rng.NextDouble();

            for (int i = 0; i < b_data.Length; i++)
                b_data[i] = (float)rng.NextDouble();

            for (int i = 0; i < c_data.Length; i++) 
                c_data[i] = (float)rng.NextDouble();

            a.Write(a_data);
            b.Write(b_data);
            c.Write(c_data);
            Matrix.TMmult(a, b, c, d);
            d.Read(d_data_gpu);

            for (int i = 0; i < N; i++)
            {
                float dot = 0;

                for (int j = 0; j < M; j++)
                {
                    dot += (a_data[i * M + j] * b_data[j]);
                }

                d_data[i] = dot * c_data[i];
            }

            var fstream = File.Open("log.txt", FileMode.Create);
            var fwriter = new StreamWriter(fstream);

            for (int n = 0; n < d_data.Length; n++)
                fwriter.Write("\t" + d_data[n] + "\t");

            fwriter.WriteLine();
            for (int n = 0; n < d_data_gpu.Length; n++)
                fwriter.Write("\t" + d_data_gpu[n] + "\t");

            fwriter.Flush();
            fwriter.Close();

            for (int w = 0; w < d_data.Length; w++)
                if (Math.Abs(d_data[w] - d_data_gpu[w]) / ((d_data[w] + d_data_gpu[w]) * 0.5f) > 0.005f)
                    Assert.Fail();
        }

        [TestMethod]
        public void VectorVectorMatrixMult()
        {
            Vector a = new Vector(M, MemoryFlags.ReadOnly, false);
            Vector b = new Vector(N, MemoryFlags.ReadOnly, false);
            Matrix c = new Matrix(N, M, MemoryFlags.WriteOnly, false);
            Random rng = new Random(0);

            float[] a_data = new float[a.Length];
            float[] b_data = new float[b.Length];
            float[] c_data = new float[c.Width * c.Height];
            float[] c_data_gpu = new float[c.Width * c.Height];

            for (int i = 0; i < a_data.Length; i++)
                a_data[i] = (float)rng.NextDouble();

            for (int i = 0; i < b_data.Length; i++)
                b_data[i] = (float)rng.NextDouble();

            a.Write(a_data);
            b.Write(b_data);
            Matrix.MatrixProduct(a, b, c);
            c.Read(c_data_gpu);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                    c_data[i * M + j] = a_data[j] * b_data[i];
            }

            var fstream = File.Open("log.txt", FileMode.Create);
            var fwriter = new StreamWriter(fstream);

            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                    fwriter.Write("\t" + c_data[n * M + m] + "\t");
                fwriter.WriteLine();
            }

            fwriter.WriteLine();
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                    fwriter.Write("\t" + c_data_gpu[n * M + m] + "\t");
                fwriter.WriteLine();
            }

            fwriter.Flush();
            fwriter.Close();

            for (int w = 0; w < c_data.Length; w++)
                if (Math.Abs(c_data[w] - c_data_gpu[w]) / ((c_data[w] + c_data_gpu[w]) * 0.5f) > 0.005f)
                    Assert.Fail();
        }

        [TestMethod]
        public void MatrixMatrixMultiplication()
        {
            Matrix a = new Matrix(K, M, MemoryFlags.ReadOnly, false);
            Matrix b = new Matrix(N, K, MemoryFlags.ReadOnly, false);

            Matrix c = new Matrix(N, M, MemoryFlags.WriteOnly, false);
            Random rng = new Random(0);

            float[] a_data = new float[a.Width * a.Height];
            float[] b_data = new float[b.Width * b.Height];

            float[] c_data = new float[c.Width * c.Height];
            float[] c_data_gpu = new float[c.Width * c.Height];

            for (int i = 0; i < a_data.Length; i++)
                a_data[i] = (float)rng.NextDouble();

            for (int i = 0; i < b_data.Length; i++)
                b_data[i] = (float)rng.NextDouble();

            a.Write(a_data);
            b.Write(b_data);
            Matrix.Multiply(a, b, c);
            c.Read(c_data_gpu);

            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++)
                        acc += a_data[k * M + m] * b_data[n * K + k];

                    c_data[n * M + m] = acc;
                }


            var fstream = File.Open("log.txt", FileMode.Create);
            var fwriter = new StreamWriter(fstream);

            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                    fwriter.Write("\t" + c_data[n * M + m] + "\t");
                fwriter.WriteLine();
            }

            fwriter.WriteLine();
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                    fwriter.Write("\t" + c_data_gpu[n * M + m] + "\t");
                fwriter.WriteLine();
            }

            fwriter.Flush();
            fwriter.Close();



            for (int w = 0; w < c_data.Length; w++)
                if (Math.Abs(c_data[w] - c_data_gpu[w]) / ((c_data[w] + c_data_gpu[w]) * 0.5f) > 0.005f)
                    Assert.Fail();
        }
    }
}
