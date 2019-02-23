using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Datasets
{
    public class UnlabeledImageSet : IDataset
    {
        private string src_dir;
        private int rsz_sz;
        private int max_imgs;
        private NRandom rng;

        private List<string> TrainingFiles;

        private Vector[] inputs;
        private bool rsz;

        public int Side { get { return rsz_sz; } }

        public UnlabeledImageSet(string directory, int resized_sz, int max_img_cnt, int seed, bool rsz)
        {
            src_dir = directory;
            rsz_sz = resized_sz;
            max_imgs = max_img_cnt;
            this.rsz = rsz;
            rng = new NRandom(seed);
        }

        public void GetNextTrainingSet(out Vector input, out Vector output)
        {
            int idx = rng.Next() % TrainingFiles.Count;
            output = input = inputs[idx];
        }

        public int GetInputSize()
        {
            return rsz_sz * rsz_sz * 3;
        }

        public int GetOutputSize()
        {
            return rsz_sz * rsz_sz * 3;
        }

        public void Initialize()
        {
            TrainingFiles = new List<string>();

            string TrainingDataPath = src_dir;
            string TrainingDataPath_SMALL = src_dir + "_small";
            int InputSize = rsz_sz * rsz_sz * 3;

            Directory.CreateDirectory(TrainingDataPath_SMALL);

            if (rsz)
            {
                var files = Directory.EnumerateFiles(src_dir).ToArray();
                Parallel.For(0, files.Length, (i) =>
                {
                    if (new string[] { ".png", ".jpg" }.Contains(Path.GetExtension(files[i])))
                    {
                        var smallPath = Path.Combine(TrainingDataPath_SMALL, Path.GetFileName(files[i]));
                        smallPath = Path.ChangeExtension(smallPath, "png");

                        if (!File.Exists(smallPath))
                        {
                            var bmp = new Bitmap(files[i]);
                            var rsz_bmp = ImageManipulation.ResizeImage(bmp, rsz_sz, rsz_sz);
                            rsz_bmp.Save(smallPath);
                            rsz_bmp.Dispose();

                            bmp.Dispose();
                        }
                    }
                });
            }
            else
            {
                TrainingDataPath_SMALL = TrainingDataPath;
            }

            if (TrainingFiles.Count == 0)
            {
                TrainingFiles.AddRange(Directory.EnumerateFiles(TrainingDataPath_SMALL));
                while (TrainingFiles.Count > max_imgs)
                {
                    TrainingFiles.RemoveAt(rng.Next() % TrainingFiles.Count);
                }
            }

            inputs = new Vector[TrainingFiles.Count];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = LoadItem(TrainingFiles[i]);
            }
        }

        public Vector LoadItem(string file)
        {
            var bmp = new Bitmap(file);
            float[] img = new float[rsz_sz * rsz_sz * 3];
            Vector img_vec = new Vector(img.Length, MemoryFlags.ReadOnly, false);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    var pxl = bmp.GetPixel(w, h);
                    img[h * bmp.Width + w] = 2.0f * pxl.R / 255.0f - 1;
                    img[bmp.Width * bmp.Height + h * bmp.Width + w] = 2.0f * pxl.G / 255.0f - 1;
                    img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] = 2.0f * pxl.B / 255.0f - 1;
                }

            bmp.Dispose();
            img_vec.Write(img);
            return img_vec;
        }
    }
}
