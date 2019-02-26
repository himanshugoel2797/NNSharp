using NNSharp.ANN.Kernels;
using OpenCL.Net;
using OpenCL.Net.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    [Serializable]
    public class Matrix : ISerializable, IDisposable
    {
        public int Columns { get; private set; }
        public int Rows { get; private set; }

        public int RowStride { get; private set; }
        public int ColumnStride { get; private set; }

#if GPU
        internal Memory memory;
#elif CPU
        internal float[] memory;
#endif

        public Matrix(int rows, int cols, MemoryFlags flags, bool zero)
        {
            Columns = cols;
            Rows = rows;

            RowStride = cols;
            ColumnStride = 1;
#if GPU
            memory = Device.GetDevice().AllocateMemory(cols * rows, flags, zero);
#elif CPU
            memory = new float[cols * rows];
#endif
        }
#if GPU
        private Matrix(Memory memory, int rows, int cols, int row_stride, int col_stride)
#elif CPU
        private Matrix(float[] memory, int rows, int cols, int row_stride, int col_stride)
#endif
        {
            this.memory = memory;
            Columns = cols;
            Rows = rows;

            RowStride = row_stride;
            ColumnStride = col_stride;
        }

        #region Bounds
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Index(int row, int col)
        {
#if INDEXING_CHECK
            if (row >= Rows | row <= 0)
                throw new Exception();

            if (col >= Columns | col <= 0)
                throw new Exception();
#endif
            return row * RowStride + col * ColumnStride;
        }

        public Matrix Reshape(int rows, int cols)
        {
            //TODO: maybe provide strides directly, to allow maintaining transposes
            //Dims:    (2,3).T = (3,2).Reshape(1,6) = (1,6)
            //Strides: (3,1)   = (1,3)              = (6,1)
            //[[1, 2, 3],[4, 5, 6]].T = [[1, 4],[2, 5],[3, 6]].R = [[1, 4, 2, 5, 3, 6]]
            //[[1, 2, 3],[4, 5, 6]].R = [[1, 2, 3, 4, 5, 6]].T = [[1],[2],[3],[4],[5],[6]]
            return new Matrix(memory, rows, cols, cols, rows);
        }

        public Matrix Transpose()
        {
            return new Matrix(memory, Columns, Rows, ColumnStride, RowStride);
        }
        #endregion

        #region Read/Write
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
        #endregion

        #region Operations
        /// <summary>
        /// O = (A dot B) + C
        /// </summary>
        /// <param name="a">NxM dimensional input matrix</param>
        /// <param name="b">MxP dimensional input matrix</param>
        /// <param name="c">Px1 dimensional optional input matrix</param>
        /// <param name="o">NxP dimensional output matrix</param>
        public static void Mad(Matrix a, Matrix b, Matrix c, Matrix o, bool reset)
        {
            if (a.Columns != b.Rows)
                throw new ArgumentException();

            if (a.Rows != o.Rows)
                throw new ArgumentException();
            if (b.Columns != o.Columns)
                throw new ArgumentException();

            if (c != null && c.Rows != b.Columns)
                throw new ArgumentException();

#if GPU
#error TODO
            //KernelManager.SGemv(a, b, false, c, KernelManager.SGemvOperation.Add, d);
#elif CPU
            Parallel.For(0, a.Rows, (i) =>
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    float acc = 0;
                    for (int k = 0; k < a.Columns; k++)
                        acc += a.memory[a.Index(i, k)] * b.memory[b.Index(k, j)];

                    if (reset)
                        o.memory[o.Index(i, j)] = acc + (c == null ? 0 : c.memory[c.Index(j, 0)]);
                    else
                        o.memory[o.Index(i, j)] += acc + (c == null ? 0 : c.memory[c.Index(j, 0)]);
                }
            });
#endif
        }

        /// <summary>
        /// C = B * rate_b + A * rate_a
        /// </summary>
        /// <param name="a">NxM dimensional optional input matrix</param>
        /// <param name="rate_a"></param>
        /// <param name="b">NxM dimensional optional input matrix</param>
        /// <param name="rate_b"></param>
        /// <param name="c">NxM dimensional output matrix</param>
        public static void Fmop(Matrix a, float rate_a, Matrix b, float rate_b, Matrix c)
        {
            if (a != null && a.Columns != c.Columns)
                throw new ArgumentException();
            if (a != null && a.Rows != c.Rows)
                throw new ArgumentException();

            if (b != null && b.Columns != c.Columns)
                throw new ArgumentException();
            if (b != null && b.Rows != c.Rows)
                throw new ArgumentException();

#if GPU
#error TODO
#elif CPU
            Parallel.For(0, c.Rows, (i) =>
            {
                for (int j = 0; j < c.Columns; j++)
                {
                    c.memory[c.Index(i, j)] = (a == null ? 0 : a.memory[a.Index(i, j)]) * rate_a + (b == null ? 0 : b.memory[b.Index(i, j)]) * rate_b;
                }
            });
#endif
        }

        /// <summary>
        /// C = activ(A) * B
        /// </summary>
        /// <param name="a">NxM dimensional input matrix</param>
        /// <param name="b">NxM dimensional optional input matrix</param>
        /// <param name="c">NxM dimensional output matrix</param>
        /// <param name="activ">Activation function to apply</param>
        public static void HadamardActivation(Matrix a, Matrix b, Matrix c, ANN.ActivationFunctionInfo activ)
        {
            if (a.Columns != c.Columns)
                throw new ArgumentException();

            if (a.Rows != c.Rows)
                throw new ArgumentException();

            if (b != null && b.Columns != c.Columns)
                throw new ArgumentException();

            if (b != null && b.Rows != c.Rows)
                throw new ArgumentException();

#if GPU
            KernelManager.HadamardActiv(a, b, c, activ);
#elif CPU
            Parallel.For(0, c.memory.Length, (i) =>
            {
                c.memory[i] = (b == null ? 0 : b.memory[i]) * activ.CPUFunction(a.memory[i]);
            });
#endif
        }
        #endregion

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
#if GPU
            var mData = new float[Width * Height];
            Read(mData);
#elif CPU
            var mData = memory;
#endif
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("width", Columns);
            info.AddValue("height", Rows);
        }

        public Matrix(SerializationInfo info, StreamingContext context)
        {
            Columns = info.GetInt32("width");
            Rows = info.GetInt32("height");

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
