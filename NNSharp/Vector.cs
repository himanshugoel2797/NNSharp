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
    public class Vector : ISerializable, IDisposable
    {
        public int Length { get; private set; }

#if GPU
        internal Memory memory;
#elif CPU
        internal float[] memory;
#endif

        public Vector(int len, MemoryFlags flags, bool zero)
        {
            Length = len;
#if GPU
            memory = Device.GetDevice().AllocateMemory(len, flags, zero);
#elif CPU
            memory = new float[len];
#endif
        }

        public void Write(float[] data)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Write(memory, data);
#elif CPU
            Array.Copy(data, memory, data.Length);
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
            var data = new float[Length];
            var dev = Device.GetDevice();
            dev.Read(memory, data);
            return data;
#elif CPU
            return memory;
#endif
        }

        public static void HadamardAct(Vector a, Vector b, Vector c, ANN.ActivationFunctionInfo activ)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            if (a.Length != c.Length)
                throw new ArgumentException();

#if GPU
            KernelManager.HadamardActiv(a, b, c, activ);
#elif CPU
            Parallel.For(0, c.memory.Length, (i) =>
            {
                c.memory[i] = a.memory[i] * activ.CPUFunction(b.memory[i]);
            });
            /*unsafe
            {
                fixed (float* a_p = a.memory)
                fixed (float* b_p = b.memory)
                fixed (float* c_p = c.memory)
                {
                    for (int i = 0; i < c.Length; i++)
                        c_p[i] = a_p[i] * Activ(activ, b_p[i]);
                }
            }*/
#endif
        }

        public static void Activation(Vector a, Vector b, ANN.ActivationFunctionInfo activ)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

#if GPU
            KernelManager.Activ(a, b, activ);
#elif CPU
            Parallel.For(0, b.memory.Length, (i) =>
            {
                b.memory[i] = activ.CPUFunction(a.memory[i]);
            });
            /*unsafe
            {
                fixed (float* a_p = a.memory)
                fixed (float* b_p = b.memory)
                {
                    for (int i = 0; i < b.Length; i++)
                        b_p[i] = Activ(activ, a_p[i]);
                }
            }*/
#endif
        }

        public static void MSubSelf(Vector a, Vector b, float rate)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

#if GPU
            //B = B - A * rate
            KernelManager.Fmop(a, -rate, b, 1);
#elif CPU
            //Parallel.For(0, b.memory.Length, (i) => b.memory[i] = b.memory[i] - a.memory[i] * rate);
            unsafe
            {
                fixed (float* a_p = a.memory)
                fixed (float* b_p = b.memory)
                {
                    for (int i = 0; i < b.Length; i++)
                        b_p[i] = b_p[i] - a_p[i] * rate;
                }
            }
#endif
        }

        public static void Mult(Vector a, float rate)
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
                    for (int i = 0; i < a.Length; i++)
                        a_p[i] *= rate;
                }
            }
#endif
        }

        public static void Add(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

#if GPU
            KernelManager.Fmop(b, 1, a, 1);
#elif CPU
            //Parallel.For(0, b.memory.Length, (i) => b.memory[i] += a.memory[i]);
            unsafe
            {
                fixed (float* a_p = a.memory)
                fixed (float* b_p = b.memory)
                {
                    for (int i = 0; i < a.Length; i++)
                        a_p[i] += b_p[i];
                }
            }
#endif
        }

        public static void Add(Vector a, Vector b, int off)
        {
#if GPU
            KernelManager.VectorConstSum(a, b, off);
#elif CPU
            //Parallel.For(0, b.memory.Length, (i) => b.memory[i] += a.memory[i]);
            unsafe
            {
                float b_v = b.memory[off];
                fixed (float* a_p = a.memory)
                {
                    for (int i = 0; i < a.Length; i++)
                        a_p[i] += b_v;
                }
            }
#endif
        }

        public static void VectorSum(Vector a, int a_off, Vector b, int b_off, int b_side)
        {
#if GPU
            KernelManager.VectorSum(a, a_off, b, b_off, b_side * b_side);
#elif CPU
            //Parallel.For(0, b.memory.Length, (i) => b.memory[i] += a.memory[i]);
            unsafe
            {
                fixed (float* a_p = a.memory)
                fixed (float* b_p = b.memory)
                {
                    for (int i = 0; i < b_side; i++)
                        for (int j = 0; j < b_side; j++)
                            a_p[a_off] += b.memory[b_off + i * b_side + j];
                }
            }
#endif
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
#if GPU
            var mData = new float[Length];
            Read(mData);
#elif CPU
            var mData = memory;
#endif
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("length", Length);
        }

        public Vector(SerializationInfo info, StreamingContext context)
        {
            Length = info.GetInt32("length");

            var mData = (float[])info.GetValue("data", typeof(float[]));
#if GPU
            var dev = Device.GetDevice();

            memory = dev.AllocateMemory(Length, MemoryFlags.ReadWrite, false);
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
        ~Vector()
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
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
