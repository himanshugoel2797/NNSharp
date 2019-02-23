using NNSharp.ANN.Layers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.OLD
{
    class ConvolutionTest
    {
        public ConvolutionTest() { }

        public static Vector LoadImage(string file)
        {
            //Bitmap bmp = new Bitmap(file);
            Vector vec = new Vector(3 * 3, MemoryFlags.ReadOnly, false);
            float[] data = new float[3 * 3];

            for (int y = 0; y < 9; y++)
                data[y] = y;

            vec.Write(data);
            return vec;
        }

        public static void SaveImage(Vector vec, string file, int s)
        {
            Bitmap bmp = new Bitmap(s, s);
            var data = vec.Read();

            for (int i = 0; i < data.Length; i++)
            {
                byte R = (byte)(data[i] * 255.0f);
                //byte G = (byte)(data[s * s + i] * 255.0f);
                //byte B = (byte)(data[2 * s * s + i] * 255.0f);

                int x = i % s;
                int y = i / s;

                bmp.SetPixel(x, y, Color.FromArgb(R, R, R));
            }

            bmp.Save(file);
            bmp.Dispose();
        }

        private float G(int x, int y, float rad)
        {
            float f = (float)(1.0f / (2 * Math.PI * 1) * Math.Exp(-(x / rad * x / rad + y / rad * y / rad) / 2));
            f = 1.0f / (rad * rad);
            return f;
        }

        public void Train()
        {
            int blur_rad = 3;

            ConvLayer testLayer = new ConvLayer();
            testLayer.SetFilterCount(1);
            testLayer.SetFilterSize(blur_rad);
            testLayer.SetInputDepth(1);
            testLayer.SetPaddingSize(0);
            testLayer.SetStrideLength(1);
            testLayer.SetInputSize(3);
            
            //Set the weights to the gaussian filter
            float[] gauss = new float[blur_rad * blur_rad];
            for (int i = 0; i < gauss.Length; i++)
            {
                int x = (i % blur_rad) - blur_rad / 2;
                int y = blur_rad / 2 - (i / blur_rad);

                //gauss[i] = G(x, y, blur_rad);
                gauss[i] = i;
            }
            testLayer.Weights[0][0].Write(gauss);
            //testLayer.Weights[0][1].Write(gauss);
            //testLayer.Weights[0][2].Write(gauss);

            //Load up an image to pass through the filter
            //testLayer.SetWeights(null);
            var res = testLayer.Forward(LoadImage(@"I:\Datasets\Lewds\conv_test.png"));
            testLayer.Error(res, true);
            SaveImage(res, "test.png", testLayer.GetFlatOutputSize()); 

            Console.WriteLine("Done.");
        }
    }
}
