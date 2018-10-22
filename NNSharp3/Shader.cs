﻿using OpenGL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3
{
    public class Shader : IDisposable
    {
        private uint program;
        private Dictionary<string, uint> parameterIdx;
        private string name = "";
#if F16
        const string f_mode = "r16f";
        const string f_single = "float16_t";
        const string f_vec4 = "f16vec4";
#else
        const string f_mode = "r32f";
        const string f_single = "float";
        const string f_vec4 = "vec4";
#endif

        public Shader(string code)
        {
            uint shader = Gl.CreateShader(ShaderType.ComputeShader);
            Gl.ShaderSource(shader, new string[] { $"#version 450 core\n#extension GL_ARB_bindless_texture : require\n#define IMG_FMT {f_mode}\n#extension GL_AMD_gpu_shader_half_float : require\n#define FLOAT_T {f_single}\n#define FLOAT4_T {f_vec4}\n" + code });
            Gl.CompileShader(shader);

            Gl.GetShader(shader, ShaderParameterName.CompileStatus, out int res);
            if (res == Gl.FALSE)
            {
                StringBuilder builder = new StringBuilder(10000);
                Gl.GetShaderInfoLog(shader, 10000, out var len, builder);
                Gl.DeleteShader(shader);

                throw new Exception(builder.ToString());
            }

            program = Gl.CreateProgram();
            Gl.AttachShader(program, shader);
            Gl.LinkProgram(program);

            /*parameterIdx = new Dictionary<string, uint>();
            for (uint i = 0; i < 256; i++)
            {
                StringBuilder buf = new StringBuilder(1000);
                Gl.GetProgramResourceName(program, ProgramInterface.Uniform, i, 1000, out var len, buf);

                if (len == 0)
                    break;

                parameterIdx[buf.ToString()] = i;
            }*/

            Gl.DeleteShader(shader);
        }

        public static Shader FromFile(string filename, params string[] defines)
        {
            string src = "";
            for (int i = 0; i < defines.Length; i++) src += defines[i] + "\n";
            src += File.ReadAllText(Path.Combine("Shaders", filename));

            return new Shader(src)
            {
                name = filename
            };
        }

        public Shader Set(string parameter, float val)
        {
            int idx = Gl.GetProgramResourceLocation(program, ProgramInterface.Uniform, parameter);
            if (idx >= 0) Gl.ProgramUniform1(program, idx, val);
            return this;
        }

        public Shader Set(string parameter, Texture val, bool read, bool write)
        {
            int rw = 0;
            if (read && write)
                rw = Gl.READ_WRITE;
            else if (read && !write)
                rw = Gl.READ_ONLY;
            else if (!read && write)
                rw = Gl.WRITE_ONLY;
            else
                throw new ArgumentException();

            //if (val.imgHandle == 0)
            {
                val.imgHandle = Gl.GetImageHandleARB(val.texID, 0, false, 0, (PixelFormat)Texture.internalFormat);
            }

            var err = Gl.GetError();
            if (err != ErrorCode.NoError) throw new Exception();
            Gl.MakeImageHandleResidentARB(val.imgHandle, rw);

            err = Gl.GetError();
            int idx = Gl.GetProgramResourceLocation(program, ProgramInterface.Uniform, parameter);
            if (idx >= 0) Gl.ProgramUniformHandleARB(program, idx, val.imgHandle);
            return this;
        }

        public void Dispatch(uint x, uint y, uint z)
        {
            Gl.MemoryBarrier(MemoryBarrierMask.AllBarrierBits);
            Gl.UseProgram(program);
            var t = DateTime.Now;
            Gl.DispatchCompute(x, y, z);
            Console.WriteLine(name + " = " + DateTime.Now.Subtract(t).Seconds);
            Gl.MemoryBarrier(MemoryBarrierMask.AllBarrierBits);
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
                if (program != 0) Gl.DeleteProgram(program);

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~Shader()
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
