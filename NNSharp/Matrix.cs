using NNSharp.ANN.Kernels;
using OpenCL.Net;
using OpenCL.Net.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    [Serializable]
    public class Matrix : ISerializable, IDisposable
    {
        public int Width { get; private set; }
        public int Height { get; private set; }

#if GPU
        internal Memory memory;
        internal MemoryFlags flags;
#elif CPU
        internal float[] memory;
#endif

        public Matrix(int w, int h, MemoryFlags flags, bool zero)
        {
            Width = w;
            Height = h;
#if GPU
            this.flags = flags;
            memory = Device.GetDevice().AllocateMemory(w * h, flags, zero);
#elif CPU
            memory = new float[w * h];
#endif
        }

        public void Write(float[] data)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Write(memory, data);
#elif CPU
            Array.Copy(data, memory, memory.Length);
#endif
        }

        public void Write(float[] data, int offset)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Write(memory, data, offset);
#elif CPU
            Array.Copy(data, 0, memory, offset, data.Length);
#endif
        }

        public void Read(float[] data)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Read(memory, data);
#elif CPU
            Array.Copy(memory, data, data.Length);
#endif
        }

        public float[] Read()
        {
#if GPU
            var data = new float[Width * Height];
            var dev = Device.GetDevice();
            dev.Read(memory, data);
            return data;
#elif CPU
            return memory;
#endif
        }

        public static void Madd(Matrix a, Vector b, Vector c, Vector d)
        {
            if (a.Width != b.Length)
                throw new ArgumentException();

            if (a.Height != c.Length)
                throw new ArgumentException();

            if (a.Height != d.Length)
                throw new ArgumentException();

#if GPU
            KernelManager.SGemv(a, b, false, c, KernelManager.SGemvOperation.Add, d);
#elif CPU
            Parallel.For(0, a.Height, (i) =>
           {
               float acc = 0;
               for (int j = 0; j < a.Width; j++)
               {
                   acc += a.memory[i + a.Height * j] * b.memory[j];
               }
               d.memory[i] = acc + c.memory[i];
           });
#endif
        }

        public static void Mult(Matrix a, float rate)
        {
#if GPU
            KernelManager.Fmop(a, 0, a, rate);
#elif CPU
            if (rate == 1)
                return;

            if (rate == 0)
            {
                Array.Clear(a.memory, 0, a.memory.Length);
                return;
            }

            //Parallel.For(0, a.memory.Length, (i) => a.memory[i] *= rate);
            unsafe
            {
                fixed (float* a_p = a.memory)
                {
                    for (int i = 0; i < a.memory.Length; i++)
                        a_p[i] *= rate;
                }
            }
#endif
        }

        public static void TMmult(Matrix a, Vector b, Vector d)
        {
            if (a.Height != b.Length)
                throw new ArgumentException();

            if (a.Width != d.Length)
                throw new ArgumentException();

#if GPU
            KernelManager.SGemv(a, b, true, null, KernelManager.SGemvOperation.None, d);
#elif CPU
            /*float[] tmp = new float[a.memory.Length];
            for (int j = 0; j < a.Width; j++)
                for (int i = 0; i < a.Height; i++)
                    tmp[j + i * a.Width] = a.memory[i + a.Height * j];

            //var tmp = a.memory;

            Parallel.For(0, a.Width, (i) =>
            {
                float acc = 0;
                for (int j = 0; j < a.Height; j++)
                {
                    acc += tmp[i + a.Width * j] * b.memory[j];
                }
                d.memory[i] = acc;
            });
            return;*/

            Parallel.For(0, a.Width, (j) =>
            {
                float acc = 0;
                for (int i = 0; i < a.Height; i++)
                {
                    acc += a.memory[i + a.Height * j] * b.memory[i];
                }
                d.memory[j] = acc;
            });
#endif
        }

        public static void MatrixProduct(Vector a, Vector b, Matrix c, bool zero)
        {
            if (a.Length != c.Height)
                throw new ArgumentException();

            if (b.Length != c.Width)
                throw new ArgumentException();

#if GPU
            KernelManager.InnerProduct(a, b, c, zero);
#elif CPU
            Parallel.For(0, b.Length, (j) =>
            {
                for (int i = 0; i < a.Length; i++)
                {
                    if (zero)
                        c.memory[i + j * c.Height] = a.memory[i] * b.memory[j];
                    else
                        c.memory[i + j * c.Height] += a.memory[i] * b.memory[j];
                }
            });
#endif
        }

        public static void MSubSelf(Matrix a, Matrix b, float rate)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

#if GPU
            //B = B - A * rate
            KernelManager.Fmop(a, -rate, b, 1);
#elif CPU
            Parallel.For(0, b.memory.Length, (i) => b.memory[i] -= a.memory[i] * rate);
#endif
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
#if GPU
            var mData = new float[Width * Height];
            Read(mData);
#elif CPU
            var mData = memory;
#endif
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("width", Width);
            info.AddValue("height", Height);
        }

        public Matrix(SerializationInfo info, StreamingContext context)
        {
            Width = info.GetInt32("width");
            Height = info.GetInt32("height");

            var mData = (float[])info.GetValue("data", typeof(float[]));

#if GPU
            var dev = Device.GetDevice();
            memory = dev.AllocateMemory(Width * Height, MemoryFlags.ReadWrite, false);
            Write(mData);
#elif CPU
            memory = mData;
#endif
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.
#if GPU
                memory.Dispose();
#elif CPU
                memory = null;
#endif

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~Matrix()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
