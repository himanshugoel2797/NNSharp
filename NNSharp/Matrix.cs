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
    public class Matrix : ISerializable
    {
        public int Width { get; private set; }
        public int Height { get; private set; }

        internal Memory memory;

        public Matrix(int w, int h, MemoryFlags flags, bool zero)
        {
            Width = w;
            Height = h;
            memory = Device.GetDevice().AllocateMemory(w * h, flags, zero);
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
            var data = new float[Width * Height];
            var dev = Device.GetDevice();
            dev.Read(memory, data);
            return data;
        }

        public static void Multiply(Matrix a, Matrix b, Matrix c)
        {
            if (a.Width != b.Height)
                throw new ArgumentException();

            if (c.Width != b.Width)
                throw new ArgumentException();

            if (c.Height != a.Height)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(c.Width);
            var optTS = Math.Min(Device.OptimalTS("gmm", c.Height, true, false), Device.OptimalTS("gmm", c.Width, true, false));
            dev["gmm", optWPT, optTS].SetArgument(c.Height)
                      .SetArgument(c.Width)
                      .SetArgument(a.Width)
                      .SetArgumentMemory(a.memory)
                      .SetArgumentMemory(b.memory)
                      .SetArgumentMemory(c.memory);

            dev.Dispatch(dev["gmm", optWPT, optTS], new uint[] { (uint)c.Height, (uint)c.Width / optWPT }, new uint[] { optTS, optTS / optWPT });
        }

        public static void Madd(Matrix a, Vector b, Vector c, Vector d)
        {
            if (a.Width != b.Length)
                throw new ArgumentException();

            if (a.Height != c.Length)
                throw new ArgumentException();

            if (a.Height != d.Length)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optTS = Device.OptimalTS("mv_madd", a.Height, false, false);
            dev["mv_madd", 1, optTS].SetArgument(a.Height)
                      .SetArgument(a.Width)
                      .SetArgumentMemory(a.memory)
                      .SetArgumentMemory(b.memory)
                      .SetArgumentMemory(c.memory)
                      .SetArgumentMemory(d.memory);

            dev.Dispatch(dev["mv_madd", 1, optTS], new uint[] { (uint)a.Height, (uint)1 }, new uint[] { optTS, 1 });
        }

        public static void TMmult(Matrix a, Vector b, Vector c, Vector d)
        {
            if (a.Height != b.Length)
                throw new ArgumentException();

            if (a.Width != c.Length)
                throw new ArgumentException();

            if (a.Width != d.Length)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optTS = Device.OptimalTS("tmv_mmult", a.Width, false, false);
            dev["tmv_mmult", 1, optTS].SetArgument(a.Height)
                      .SetArgument(a.Width)
                      .SetArgumentMemory(a.memory)
                      .SetArgumentMemory(b.memory)
                      .SetArgumentMemory(c.memory)
                      .SetArgumentMemory(d.memory);

            dev.Dispatch(dev["tmv_mmult", 1, optTS], new uint[] { (uint)a.Width, (uint)1 }, new uint[] { optTS, 1 });
        }

        public static void MatrixProduct(Vector a, Vector b, Matrix c)
        {
            if (a.Length != c.Height)
                throw new ArgumentException();

            if (b.Length != c.Width)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optTS = Math.Min(Device.OptimalTS("vv_mmult", a.Length, true, false), Device.OptimalTS("vv_mmult", b.Length, true, false));
            dev["vv_mmult", 1, optTS].SetArgument(a.Length)
                           .SetArgument(b.Length)
                           .SetArgumentMemory(a.memory)
                           .SetArgumentMemory(b.memory)
                           .SetArgumentMemory(c.memory);

            dev.Dispatch(dev["vv_mmult", 1, optTS], new uint[] { (uint)a.Length, (uint)b.Length }, new uint[] { optTS, optTS });
        }

        public static void MSub(Matrix a, Matrix b, float rate, Matrix c)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            var dev = Device.GetDevice();

            var optWPT = Device.OptimalWPT(a.Height * a.Width);
            var optTS = Device.OptimalTS("v_msub", a.Height * a.Width, false, true);
            dev["v_msub", optWPT, optTS].SetArgument(rate)
                         .SetArgumentMemory(a.memory)
                         .SetArgumentMemory(b.memory)
                         .SetArgumentMemory(c.memory);

            dev.Dispatch(dev["v_msub", optWPT, optTS], new uint[] { (uint)a.Height * (uint)a.Width / optWPT, 1 }, new uint[] { optTS, 1 });
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            var mData = new float[Width * Height];
            Read(mData);
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("width", Width);
            info.AddValue("height", Height);
        }

        public Matrix(SerializationInfo info, StreamingContext context)
        {
            Width = info.GetInt32("width");
            Height = info.GetInt32("height");

            var mData = (float[])info.GetValue("data", typeof(float[]));
            var dev = Device.GetDevice();

            memory = dev.AllocateMemory(Width * Height, MemoryFlags.ReadWrite, false);
            Write(mData);
        }
    }
}
