using NNSharp3.Math;
using OpenGL;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3
{
    public class Texture : IDisposable
    {
        internal uint texID;
        internal uint bufferID;
        internal ulong imgHandle;

        private int w, h;

        public int Width { get { return w; } }
        public int Height { get { return h; } }

#if F16
        internal const InternalFormat internalFormat = InternalFormat.R16f;
        const int Multiplier = 2;
#else
        internal const InternalFormat internalFormat = InternalFormat.R32f;
        const int Multiplier = 4;
#endif

        public Texture(int w, int h)
        {
            this.w = w;
            this.h = h;

            texID = Gl.CreateTexture((TextureTarget)Gl.TEXTURE_BUFFER);
            bufferID = Gl.CreateBuffer();
            Gl.NamedBufferStorage(bufferID, (uint)(w * h * Multiplier), IntPtr.Zero, Gl.DYNAMIC_STORAGE_BIT);
            Gl.TextureBuffer(texID, internalFormat, bufferID);
        }

        public void SetData(Bitmap bmp)
        {
#if F16
            var d = new Half[bmp.Width * bmp.Height * 4];
            for(int y = 0; y < bmp.Height; y++)
                for(int x = 0; x < bmp.Width; x++)
                {
                    d[(y * bmp.Width + x) * 4] = new Half(bmp.GetPixel(x, y).R / (float)255);
                    d[(y * bmp.Width + x) * 4 + 1] = new Half(bmp.GetPixel(x, y).G / (float)255);
                    d[(y * bmp.Width + x) * 4 + 2] = new Half(bmp.GetPixel(x, y).B / (float)255);
                    d[(y * bmp.Width + x) * 4 + 3] = new Half(bmp.GetPixel(x, y).A / (float)255);

                }
#else
            var d = new float[bmp.Width * bmp.Height * 4];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    d[(y * bmp.Width + x) * 4] = (bmp.GetPixel(x, y).R / (float)255);
                    d[(y * bmp.Width + x) * 4 + 1] = (bmp.GetPixel(x, y).G / (float)255);
                    d[(y * bmp.Width + x) * 4 + 2] = (bmp.GetPixel(x, y).B / (float)255);
                    d[(y * bmp.Width + x) * 4 + 3] = (bmp.GetPixel(x, y).A / (float)255);

                }
#endif
            Gl.NamedBufferSubData(bufferID, IntPtr.Zero, (uint)(bmp.Width * bmp.Height * 4 * Multiplier), d);
        }

        public void SetData(byte[] data)
        {
#if F16
            var d = new Half[data.Length];
            for (int i = 0; i < d.Length; i++)
                d[i] = new Half(data[i]);
#else
            var d = new float[data.Length];
            for (int i = 0; i < d.Length; i++)
                d[i] = (data[i]);
#endif
            Gl.NamedBufferSubData(bufferID, IntPtr.Zero, (uint)(data.Length * Multiplier), d);
        }

        public void SetData(float[] data)
        {
#if F16
            var d = new Half[data.Length];
            for (int i = 0; i < d.Length; i++)
                d[i] = new Half(data[i]);
#else
            var d = data;
#endif
            Gl.NamedBufferSubData(bufferID, IntPtr.Zero, (uint)(data.Length * Multiplier), d);
        }

        public void GetData(float[] data)
        {
#if F16
            var d = new Half[data.Length];
#else
            var d = data;
#endif
            Gl.GetNamedBufferSubData(bufferID, IntPtr.Zero, (uint)(data.Length * Multiplier), d);
#if F16
            for (int i = 0; i < data.Length; i++)
                data[i] = d[i];
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
                if (bufferID != 0) Gl.DeleteBuffers(bufferID);
                if (texID != 0) Gl.DeleteTextures(texID);

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~Texture()
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
