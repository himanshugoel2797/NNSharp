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
        public float[] Memory;
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
            Memory = new float[cols * rows];
#endif
        }
#if GPU
        private Matrix(Memory memory, int rows, int cols, int row_stride, int col_stride)
#elif CPU
        private Matrix(float[] memory, int rows, int cols, int row_stride, int col_stride)
#endif
        {
            this.Memory = memory;
            Columns = cols;
            Rows = rows;

            RowStride = row_stride;
            ColumnStride = col_stride;

            RowStride = cols;
            ColumnStride = 1;
        }

        #region Bounds
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Index(int row, int col)
        {
#if INDEXING_CHECK
            if (row >= Rows | row < 0)
                throw new Exception();

            if (col >= Columns | col < 0)
                throw new Exception();
#endif
            return row * RowStride + col * ColumnStride;
        }

        public Matrix Reshape(int rows, int cols)
        {
            if (rows * cols != Rows * Columns) throw new Exception();
            //TODO: maybe provide strides directly, to allow maintaining transposes
            //Dims:    (2,3).T = (3,2).Reshape(1,6) = (1,6)
            //Strides: (3,1)   = (1,3)              = (6,1)
            //[[1, 2, 3],[4, 5, 6]].T = [[1, 4],[2, 5],[3, 6]].R = [[1, 4, 2, 5, 3, 6]]
            //[[1, 2, 3],[4, 5, 6]].R = [[1, 2, 3, 4, 5, 6]].T = [[1],[2],[3],[4],[5],[6]]
            return new Matrix(Memory, rows, cols, cols, rows);
        }

        public Matrix Transpose()
        {
            return new Matrix(Memory, Columns, Rows, ColumnStride, RowStride);
        }
        #endregion

        #region Read/Write
        public void Write(float[] data)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Write(memory, data);
#elif CPU
            Array.Copy(data, Memory, Memory.Length);
#endif
        }

        public void Write(float[] data, int offset)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Write(memory, data, offset);
#elif CPU
            Array.Copy(data, 0, Memory, offset, data.Length);
#endif
        }

        public void Read(float[] data)
        {
#if GPU
            var dev = Device.GetDevice();
            dev.Read(memory, data);
#elif CPU
            Array.Copy(Memory, data, data.Length);
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
            return Memory;
#endif
        }
        #endregion

        #region Operations
        /// <summary>
        /// O = (A dot B) + C + D
        /// </summary>
        /// <param name="a">NxM dimensional input matrix</param>
        /// <param name="b">MxP dimensional input matrix</param>
        /// <param name="c">Px1 dimensional optional input matrix</param>
        /// <param name="d">Nx1 dimensional optional input matrix</param>
        /// <param name="o">NxP dimensional output matrix</param>
        public static void Mad(Matrix a, Matrix b, Matrix c, Matrix d, Matrix o, bool reset)
        {
            if (a.Columns != b.Rows)
                throw new ArgumentException();

            if (a.Rows != o.Rows)
                throw new ArgumentException();
            if (b.Columns != o.Columns)
                throw new ArgumentException();

            if (c != null && c.Rows != b.Columns)
                throw new ArgumentException();

            if (d != null && d.Rows != a.Rows)
                throw new ArgumentException();

#if GPU
#error TODO
            //KernelManager.SGemv(a, b, false, c, KernelManager.SGemvOperation.Add, d);
#elif CPU
            Parallel.For(0, a.Rows, (i) =>
            //for(int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    float acc = 0;
                    for (int k = 0; k < a.Columns; k++)
                    {
                        //TODO: Check if either one is continuously indexed, if so, use a version that uses and increments pointers directly
                        acc += a.Memory[a.Index(i, k)] * b.Memory[b.Index(k, j)];
                    }

                    if (reset)
                        o.Memory[o.Index(i, j)] = acc + (c == null ? 0 : c.Memory[c.Index(j, 0)]) + (d == null ? 0 : d.Memory[d.Index(i, 0)]);
                    else
                        o.Memory[o.Index(i, j)] += acc + (c == null ? 0 : c.Memory[c.Index(j, 0)]) + (d == null ? 0 : d.Memory[d.Index(i, 0)]);
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
            if (c.ColumnStride == 1 && ((a != null && a.ColumnStride == 1) | a == null) && ((b != null && b.ColumnStride == 1) | b == null))
            {
                unsafe
                {
                    fixed (float* a_p_b = a?.Memory)
                    fixed (float* b_p_b = b?.Memory)
                    fixed (float* c_p_b = c.Memory)
                    {
                        float* a_p = a_p_b;
                        float* b_p = b_p_b;
                        float* c_p = c_p_b;

                        for (int i = 0; i < c.Memory.Length; i++, a_p++, b_p++, c_p++)
                            *c_p = (a == null ? 0 : *a_p) * rate_a + (b == null ? 0 : *b_p) * rate_b;
                    }
                }
            }
            else
                Parallel.For(0, c.Rows, (i) =>
                {
                    for (int j = 0; j < c.Columns; j++)
                    {
                        c.Memory[c.Index(i, j)] = (a == null ? 0 : a.Memory[a.Index(i, j)]) * rate_a + (b == null ? 0 : b.Memory[b.Index(i, j)]) * rate_b;
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
            Parallel.For(0, c.Memory.Length, (i) =>
            {
                c.Memory[i] = (b == null ? 1 : b.Memory[i]) * activ.CPUFunction(a.Memory[i]);
            });
#endif
        }

        /// <summary>
        /// Extract blocks of filter_sz from the input and convert them into columns
        /// </summary>
        /// <param name="input_sz"></param>
        /// <param name="input_cnt"></param>
        /// <param name="stride_len"></param>
        /// <param name="padding"></param>
        /// <param name="filter_sz"></param>
        /// <param name="output_sz"></param>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public static void Image2Column(int input_sz, int input_cnt, int stride_len, int padding, int filter_sz, int output_sz, Matrix input, Matrix output)
        {
            int block_sz = filter_sz * filter_sz * input_cnt;   //output rows
            int len = output_sz * output_sz;                    //output columns

            //for (int i_col = 0; i_col < len; i_col++)
            Parallel.For(0, len, (i_col) =>
            {
                for (int i_d = 0; i_d < input_cnt; i_d++)
                    for (int f_row = 0; f_row < filter_sz; f_row++)
                        for (int f_col = 0; f_col < filter_sz; f_col++)
                        {
                            int f_row0 = (i_col / output_sz) * stride_len + f_row - padding;
                            int f_col0 = (i_col % output_sz) * stride_len + f_col - padding;

                            output.Memory[output.Index(i_d * filter_sz * filter_sz + f_row * filter_sz + f_col, i_col)] = 0;

                            if (f_row0 >= 0 && f_col0 >= 0 && f_row0 < input_sz && f_col0 < input_sz)
                                output.Memory[output.Index(i_d * filter_sz * filter_sz + f_row * filter_sz + f_col, i_col)] = input.Memory[input.Index(i_d, f_row0 * input_sz + f_col0)];
                        }
            });
        }

        public static void Column2Image(int input_sz, int input_cnt, int stride_len, int padding, int filter_sz, int output_sz, Matrix input, Matrix output, Matrix inc_cnt = null)
        {
            //Foreach column in input, rearrange it into a block, stripping padding and applying appropriate strides
            int block_sz = filter_sz * filter_sz * input_cnt;
            int len = output_sz * output_sz;

            if (inc_cnt == null) inc_cnt = new Matrix(input.Rows, input.Columns, MemoryFlags.ReadWrite, true);

            input.Clear();
            /*
            for (int i_col = 0; i_col < len; i_col++)
                for (int i_d = 0; i_d < input_cnt; i_d++)
                    for (int f_row = 0; f_row < filter_sz; f_row++)
                        for (int f_col = 0; f_col < filter_sz; f_col++)
                        {
                            int f_row0 = (i_col / output_sz) * stride_len + f_row - padding;
                            int f_col0 = (i_col % output_sz) * stride_len + f_col - padding;

                            if (f_row0 >= 0 && f_col0 >= 0 && f_row0 < input_sz && f_col0 < input_sz)
                                input.memory[input.Index(i_d, f_row0 * input_sz + f_col0)] += output.memory[output.Index(i_d * filter_sz * filter_sz + f_row * filter_sz + f_col, i_col)];
                        }*/
            //for (int c = 0; c < block_sz; c++)
            Parallel.For(0, block_sz, (c) =>
            {
                int col_off = (c % filter_sz);
                int row_off = (c / filter_sz) % filter_sz;
                int c_im = c / filter_sz / filter_sz;
                for (int row = 0; row < output_sz; row++)
                    for (int col = 0; col < output_sz; col++)
                    {
                        int row_pad = row * stride_len - padding + row_off;
                        int col_pad = col * stride_len - padding + col_off;
                        if (row_pad >= 0 && row_pad < input_sz && col_pad >= 0 && col_pad < input_sz)
                        {
                            input.Memory[input.Index(c_im, row_pad * input_sz + col_pad)] += output.Memory[c * output_sz * output_sz + row * output_sz + col];
                            inc_cnt.Memory[input.Index(c_im, row_pad * input_sz + col_pad)]++;
                        }
                    }
            });

            //for (int i = 0; i < input_sz * input_sz; i++)
            /*Parallel.For(0, input_sz * input_sz, (i) =>
            {
                input.memory[i] /= inc_cnt.memory[i];
            });*/
        }

        public static void Convolve(Matrix input, bool rotInput, int inputOff, int inputSz, int paddingSz, int dilation, int strideLen, Matrix filter, bool rotFilter, int filterOff, int filterSz, Matrix output, bool rotOutput, int outputOff, int outputSz, bool zero, Matrix bias = null, int bias_off = 0)
        {
            if (zero)
                output.Clear();

            unsafe
            {
                fixed (float* filter_m = &filter.Memory[filterOff])
                fixed (float* input_m = &input.Memory[inputOff])
                fixed (float* output_m = &output.Memory[outputOff])
                {

                    if (filterSz < outputSz)
                    {
                        int f_sz = filterSz * filterSz - 1;
                        for (int c = 0; c < filterSz * filterSz; ++c)
                        {
                            int x0 = c % filterSz;
                            int y0 = c / filterSz;
                            float filter_val = filter_m[(rotFilter ? c : f_sz - c)];

                            for (int y = 0; y < outputSz; y++)
                            {
                                int i_y = y * strideLen + y0 * dilation - paddingSz;
                                if (rotInput) i_y = inputSz - 1 - i_y;

                                if (bias != null)
                                {
                                    float b_val = bias.Memory[bias_off];
                                    for (int x = 0; x < outputSz; x++)
                                    {
                                        if (rotOutput) output_m[(outputSz - 1 - y) * outputSz + (outputSz - 1 - x)] += b_val;
                                        else output_m[y * outputSz + x] += b_val;
                                    }
                                }

                                if (i_y >= 0 && i_y < inputSz)
                                    for (int x = 0; x < outputSz; x++)
                                    {
                                        int i_x = x * strideLen + x0 * dilation - paddingSz;
                                        if (rotInput) i_x = inputSz - 1 - i_x;

                                        if (i_x >= 0 && i_x < inputSz)
                                        {
                                            float output_val = filter_val * input_m[i_y * inputSz + i_x];

                                            if (rotOutput) output_m[(outputSz - 1 - y) * outputSz + (outputSz - 1 - x)] += output_val;
                                            else output_m[y * outputSz + x] += output_val;
                                        }
                                    }
                            }
                        }
                    }
                    else
                    {
                        int o_sz = outputSz * outputSz - 1;
                        for (int c = 0; c < outputSz * outputSz; ++c)
                        {
                            int x = c % outputSz;
                            int y = c / outputSz;
                            float output_val = 0;

                            if (bias != null)
                                output_val += bias.Memory[bias_off];

                            for (int y0 = 0; y0 < filterSz; y0++)
                            {
                                int i_y = y * strideLen + y0 * dilation - paddingSz;
                                if (rotInput) i_y = inputSz - 1 - i_y;

                                if (i_y >= 0 && i_y < inputSz)
                                    for (int x0 = 0; x0 < filterSz; x0++)
                                    {
                                        int i_x = x * strideLen + x0 * dilation - paddingSz;
                                        if (rotInput) i_x = inputSz - 1 - i_x;

                                        float filter_val = filter_m[(filterSz - 1 - y0) * filterSz + (filterSz - 1 - x0)];
                                        if (rotFilter) filter_val = filter_m[y0 * filterSz + x0];

                                        if (i_x >= 0 && i_x < inputSz)
                                            output_val += filter_val * input_m[i_y * inputSz + i_x];
                                    }
                            }

                            output_m[(rotOutput ? o_sz - c : c)] += output_val;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Clear the matrix.
        /// </summary>
        public void Clear()
        {
#if GPU
#error TODO
#elif CPU
            Array.Clear(Memory, 0, Memory.Length);
#endif
        }
        #endregion

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
#if GPU
            var mData = new float[Width * Height];
            Read(mData);
#elif CPU
            var mData = Memory;
#endif
            info.AddValue("data", mData, mData.GetType());
            info.AddValue("width", Columns);
            info.AddValue("height", Rows);
            info.AddValue("width_stride", ColumnStride);
            info.AddValue("height_stride", RowStride);
        }

        public Matrix(SerializationInfo info, StreamingContext context)
        {
            Columns = info.GetInt32("width");
            Rows = info.GetInt32("height");
            ColumnStride = info.GetInt32("width_stride");
            RowStride = info.GetInt32("height_stride");

            var mData = (float[])info.GetValue("data", typeof(float[]));

#if GPU
            var dev = Device.GetDevice();
            memory = dev.AllocateMemory(Width * Height, MemoryFlags.ReadWrite, false);
            Write(mData);
#elif CPU
            Memory = mData;
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
                Memory = null;
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
