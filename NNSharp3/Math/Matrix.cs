using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.Math
{
    public class Matrix
    {
        public int Rows { get; private set; }
        public int Columns { get; private set; }

        internal Texture tex;
        static Shader matrix_init = null;

        static Matrix()
        {
            matrix_init = Shader.FromFile("constant_memory_init.glsl");
        }

        public Matrix(int rows, int columns, float iVal = 0)
        {
            Rows = rows;
            Columns = columns;

            //TODO: Allocate a texture
            tex = new Texture(columns, rows);
            //TODO: Run a shader to fill the texture with iVal
            matrix_init.Set("val", iVal);
            matrix_init.Set("cols_cnt", columns);
            matrix_init.Set("w", tex, false, true);
            matrix_init.Dispatch((uint)tex.Width, (uint)tex.Height, 1);

            matrix_init.Dispatch((uint)tex.Width, (uint)tex.Height, 1);
        }

        public Matrix(Texture tex, int rows, int cols)
        {
            this.tex = tex;
            Rows = rows;
            Columns = cols;
        }

        public void Write(float[] data)
        {
            tex.SetData(data);
        }

        public void Read(float[] data)
        {
            tex.GetData(data);
        }
    }
}
