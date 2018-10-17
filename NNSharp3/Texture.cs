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
        internal ulong imgHandle;

        private int w, h;

        public int Width { get { return w; } }
        public int Height { get { return h; } }

        public Texture(int w, int h)
        {
            if (w % 4 != 0)
                w += 4 - w % 4;

            this.w = w / 4;
            this.h = h;

            texID = Gl.CreateTexture(TextureTarget.Texture2d);
            Gl.TextureStorage2D(texID, 1, InternalFormat.Rgba16f, w / 4, h);

        }

        public void SetData(Bitmap bmp)
        {
            IntPtr pval = IntPtr.Zero;
            System.Drawing.Imaging.BitmapData bd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            try
            {
                pval = bd.Scan0;
                Gl.TextureSubImage2D(texID, 0, 0, 0, w, h, OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, pval);
            }
            finally
            {
                bmp.UnlockBits(bd);
            }
        }

        public void SetData(byte[] data)
        {
            Gl.TextureSubImage2D(texID, 0, 0, 0, w, h, OpenGL.PixelFormat.Rgba, PixelType.HalfFloat, data);
        }

        public void SetData(float[] data)
        {
            Gl.TextureSubImage2D(texID, 0, 0, 0, w, h, OpenGL.PixelFormat.Rgba, PixelType.Float, data);
        }

        public void GetData(float[] data)
        {
            Gl.GetTextureImage(texID, 0, OpenGL.PixelFormat.Rgba, PixelType.Float, data.Length * sizeof(float), data);
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
