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

        public float[] Memory;
        internal Memory GPUMemory;
        private bool GPU_uptodate;
        private bool CPU_uptodate;

        public Matrix(int rows, int cols, MemoryFlags flags, bool zero)
        {
            Columns = cols;
            Rows = rows;

            RowStride = cols;
            ColumnStride = 1;
            Memory = new float[cols * rows];

            GPU_uptodate = false;
            CPU_uptodate = true;
        }

        private Matrix(float[] memory, Memory gpu_mem, bool gpu_u2d, bool cpu_u2d, int rows, int cols, int row_stride, int col_stride)
        {
            Memory = memory;
            GPUMemory = gpu_mem;

            Columns = cols;
            Rows = rows;

            RowStride = row_stride;
            ColumnStride = col_stride;

            RowStride = cols;
            ColumnStride = 1;

            GPU_uptodate = gpu_u2d;
            CPU_uptodate = cpu_u2d;
        }

        #region GPU Management
        private void UpdateGPU()
        {
            var dev = Device.GetDevice();
            if (GPUMemory == null) GPUMemory = dev.AllocateMemory(Memory.Length, MemoryFlags.ReadWrite, false);
            dev.Write(GPUMemory, Memory);
            GPU_uptodate = true;
        }

        private void UpdateCPU()
        {
            if (GPUMemory != null)
            {
                var dev = Device.GetDevice();
                dev.Read(GPUMemory, Memory);
            }
            CPU_uptodate = true;
        }
        #endregion

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
            return new Matrix(Memory, GPUMemory, GPU_uptodate, CPU_uptodate, rows, cols, cols, rows);
        }

        public Matrix Transpose()
        {
            return new Matrix(Memory, GPUMemory, GPU_uptodate, CPU_uptodate, Columns, Rows, ColumnStride, RowStride);
        }
        #endregion

        #region Read/Write
        public void Write(float[] data)
        {
            Array.Copy(data, Memory, Memory.Length);
        }

        public void Write(float[] data, int offset)
        {
            Array.Copy(data, 0, Memory, offset, data.Length);
        }

        public void Read(float[] data)
        {
            Array.Copy(Memory, data, data.Length);
        }

        public float[] Read()
        {
            return Memory;
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

            if (!a.CPU_uptodate)
                a.UpdateCPU();
            if (!b.CPU_uptodate)
                b.UpdateCPU();
            if (c != null && !c.CPU_uptodate)
                c.UpdateCPU();
            if (d != null && !d.CPU_uptodate)
                d.UpdateCPU();

            if (a.Rows == 1 && a.Columns == 1 && b.Rows == 1 && b.Columns == 1)
            {
                if (reset)
                    o.Memory[0] = a.Memory[0] * b.Memory[0] + (c == null ? 0 : c.Memory[0]) + (d == null ? 0 : d.Memory[0]);
                else
                    o.Memory[0] += a.Memory[0] * b.Memory[0] + (c == null ? 0 : c.Memory[0]) + (d == null ? 0 : d.Memory[0]);
            }
            else if (a.Rows == 1)
            {
                Parallel.For(0, b.Columns, (j) =>
                {
                    float acc = 0;

                    unsafe
                    {
                        fixed (float* a_p = a.Memory)
                        {
                            float* a_pi = a_p;
                            for (int i = 0; i < b.Rows; i++)
                                acc += b.Memory[b.Index(i, j)] * *(a_pi++);
                        }
                    }

                    if (reset)
                        o.Memory[j] = acc + (c == null ? 0 : c.Memory[c.Index(j, 0)]) + (d == null ? 0 : d.Memory[0]);
                    else
                        o.Memory[j] += acc + (c == null ? 0 : c.Memory[c.Index(j, 0)]) + (d == null ? 0 : d.Memory[0]);
                });
            }
            else
            {
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
            }

            o.GPU_uptodate = false;
            o.CPU_uptodate = true;
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


            if (b != null && !b.CPU_uptodate)
                b.UpdateCPU();
            if (a != null && !a.CPU_uptodate)
                a.UpdateCPU();

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

            c.GPU_uptodate = false;
            c.CPU_uptodate = true;
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

            if (b != null && !b.CPU_uptodate)
                b.UpdateCPU();
            if (!a.CPU_uptodate)
                a.UpdateCPU();

            Parallel.For(0, c.Memory.Length, (i) =>
            {
                c.Memory[i] = (b == null ? 1 : b.Memory[i]) * activ.CPUFunction(a.Memory[i]);
            });

            c.GPU_uptodate = false;
            c.CPU_uptodate = true;
        }

        public static void Convolve(Matrix input, bool rotInput, int inputOff, int inputSz, int paddingSz, int dilation, float strideLen, Matrix filter, bool rotFilter, int filterOff, int filterSz, Matrix output, bool rotOutput, int outputOff, int outputSz, bool zero, Matrix bias = null, int bias_off = 0)
        {
            if (KernelManager.GPUMode)
            {
                if (!input.GPU_uptodate)
                    input.UpdateGPU();
                if (!filter.GPU_uptodate)
                    filter.UpdateGPU();
                if (!output.GPU_uptodate)
                    output.UpdateGPU();

                KernelManager.Convolve(input, inputOff, inputSz, filter, filterOff, filterSz, rotFilter, paddingSz, strideLen, output, outputOff, outputSz, rotOutput, zero, bias, bias_off);

                output.CPU_uptodate = false;
                output.GPU_uptodate = true;

                return;
            }

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
                                int i_y = (int)(y * strideLen + y0 * dilation - paddingSz);
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
                                        int i_x = (int)(x * strideLen + x0 * dilation - paddingSz);
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
                                int i_y = (int)(y * strideLen + y0 * dilation - paddingSz);
                                if (rotInput) i_y = inputSz - 1 - i_y;

                                if (i_y >= 0 && i_y < inputSz)
                                    for (int x0 = 0; x0 < filterSz; x0++)
                                    {
                                        int i_x = (int)(x * strideLen + x0 * dilation - paddingSz);
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
            Array.Clear(Memory, 0, Memory.Length);
            if (GPUMemory == null) UpdateGPU();
            //Device.GetDevice().Fill(GPUMemory, 0, Memory.Length, 0);
            CPU_uptodate = true;
            GPU_uptodate = true;
        }
        #endregion

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            var mData = Memory;
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
            Memory = mData;
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
                Memory = null;
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
