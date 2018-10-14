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

        internal Memory memory;
        
        public Vector(int len, MemoryFlags flags, bool zero)
        {
            Length = len;
            memory = Device.GetDevice().AllocateMemory(len, flags, zero);
        }
        
        public void Write(float[] data)
        {
            var dev = Device.GetDevice();
            dev.Write(memory, data);
        }

        public void Write(float[] data, int offset)
        {
            var dev = Device.GetDevice();
            dev.Write(memory, data, offset);
        }

        public void Read(float[] data)
        {
            var dev = Device.GetDevice();
            dev.Read(memory, data);
        }

        public float[] Read()
        {
            var data = new float[Length];
            var dev = Device.GetDevice();
            dev.Read(memory, data);
            return data;
        }

        public static void Hadamard(Vector a, Vector b, Vector c)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            if (a.Length != c.Length)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(a.Length);
            var optTS = Device.OptimalTS("v_hadamard", a.Length, false, true);
            dev["v_hadamard", optWPT, optTS].SetArgumentMemory(a.memory)
                             .SetArgumentMemory(b.memory)
                             .SetArgumentMemory(c.memory);

            dev.Dispatch(dev["v_hadamard", optWPT, optTS], new uint[] { (uint)a.Length / optWPT, (uint)1 }, new uint[] { optTS, 1 });
        }

        public static void MSub(Vector a, Vector b, float rate, Vector c)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(a.Length);
            var optTS = Device.OptimalTS("v_msub", a.Length, false, true);
            dev["v_msub", optWPT, optTS].SetArgument(rate)
                           .SetArgumentMemory(a.memory)
                           .SetArgumentMemory(b.memory)
                           .SetArgumentMemory(c.memory);

            dev.Dispatch(dev["v_msub", optWPT, optTS], new uint[] { (uint)a.Length / optWPT, 1 }, new uint[] { optTS, 1 });
        }

        public static void Add(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(a.Length);
            var optTS = Device.OptimalTS("v_add", a.Length, false, true);
            dev["v_add", optWPT, optTS].SetArgumentMemory(a.memory)
                         .SetArgumentMemory(b.memory);

            dev.Dispatch(dev["v_add", optWPT, optTS], new uint[] { (uint)a.Length / optWPT, 1 }, new uint[] { optTS, 1 });
        }

        public static void Divide(float a, Vector b)
        {
            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(b.Length);
            var optTS = Device.OptimalTS("v_div", b.Length, false, true);
            dev["v_div", optWPT, optTS].SetArgument(a)
                         .SetArgumentMemory(b.memory);

            dev.Dispatch(dev["v_div", optWPT, optTS], new uint[] { (uint)b.Length / optWPT, 1 }, new uint[] { optTS, 1 });
        }


        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            var mData = new float[Length];
            Read(mData);
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("length", Length);
        }

        public Vector(SerializationInfo info, StreamingContext context)
        {
            Length = info.GetInt32("length");

            var mData = (float[])info.GetValue("data", typeof(float[]));
            var dev = Device.GetDevice();

            memory = dev.AllocateMemory(Length, MemoryFlags.ReadWrite, false);
            Write(mData);
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
                memory.Dispose();

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
