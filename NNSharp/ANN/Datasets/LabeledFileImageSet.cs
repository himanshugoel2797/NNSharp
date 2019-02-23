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
    public class LabeledFileImageSet : IDataset
    {
        private int rsz_sz, tag_cnt, chnl_cnt;
        private int max_imgs;
        private NRandom rng;
        private string TrainingDataPath_SMALL;
        private List<string> TrainingFiles;
        private List<float[]> TrainingTags;

        private Vector[] inputs, outputs;

        public int Side { get { return rsz_sz; } }

        public LabeledFileImageSet(string small_path, int resized_sz, int max_img_cnt, int tag_cnt, int chnl_cnt, int seed)
        {
            rsz_sz = resized_sz;
            this.tag_cnt = tag_cnt;
            this.chnl_cnt = chnl_cnt;
            max_imgs = max_img_cnt;
            TrainingDataPath_SMALL = small_path;
            rng = new NRandom(seed);

            TrainingFiles = new List<string>();
            TrainingTags = new List<float[]>();
        }

        public void GetNextTrainingSet(out Vector input, out Vector output)
        {
            int idx = rng.Next() % TrainingFiles.Count;
            input = inputs[idx];
            output = outputs[idx];
        }

        public int GetInputSize()
        {
            return rsz_sz * rsz_sz * chnl_cnt;
        }

        public int GetOutputSize()
        {
            return tag_cnt;
        }

        public void AddFile(string file, float[] tags)
        {
            if (tags.Length != tag_cnt) throw new Exception();
            TrainingFiles.Add(file);
            TrainingTags.Add(tags);
        }

        public void Initialize()
        {
            int InputSize = rsz_sz * rsz_sz * chnl_cnt;

            Directory.CreateDirectory(TrainingDataPath_SMALL);

            object l_obj = new object();
            List<string> nFiles = new List<string>();
            List<float[]> nTags = new List<float[]>();
            Parallel.For(0, TrainingFiles.Count, (i) =>
            {
                var smallPath = Path.Combine(TrainingDataPath_SMALL, i + ".png");

                if (!File.Exists(smallPath))
                {
                    var bmp = new Bitmap(TrainingFiles[i]);

                    int w = rsz_sz * 3;
                    int h = rsz_sz * 3 * bmp.Height / bmp.Width;

                    if (bmp.Width > bmp.Height)
                    {
                        h = rsz_sz * 3;
                        w = rsz_sz * 3 * bmp.Width / bmp.Height;
                    }

                    var rsz_bmp = ImageManipulation.CutImage(ImageManipulation.ResizeImage(bmp, w, h), rsz_sz, rsz_sz, TrainingDataPath_SMALL, i.ToString());
                    bmp.Dispose();

                    lock (l_obj)
                    {
                        nFiles.AddRange(rsz_bmp);
                        for (int j = 0; j < rsz_bmp.Length; j++)
                            nTags.Add(TrainingTags[i]);
                    }
                }
                else
                {
                    lock (l_obj)
                    {
                        nFiles.Add(smallPath);
                        nTags.Add(TrainingTags[i]);
                    }
                }
            });

            TrainingFiles = nFiles;
            TrainingTags = nTags;

            inputs = new Vector[TrainingFiles.Count];
            outputs = new Vector[TrainingFiles.Count];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = LoadItem(TrainingFiles[i]);
                outputs[i] = new Vector(tag_cnt, MemoryFlags.ReadOnly, false);
                outputs[i].Write(TrainingTags[i]);
            }

            //Swap these data around
            for (int i = 0; i < inputs.Length / 2; i++)
            {
                int src_idx = rng.Next() % inputs.Length;
                int dst_idx = rng.Next() % inputs.Length;

                var tmp_i = inputs[src_idx];
                var tmp_o = outputs[src_idx];

                inputs[src_idx] = inputs[dst_idx];
                outputs[src_idx] = outputs[dst_idx];

                inputs[dst_idx] = tmp_i;
                outputs[dst_idx] = tmp_o;
            }
        }

        public Vector LoadItem(string file)
        {
            var bmp = new Bitmap(file);
            float[] img = new float[rsz_sz * rsz_sz * chnl_cnt];
            Vector img_vec = new Vector(img.Length, MemoryFlags.ReadOnly, false);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    var pxl = bmp.GetPixel(w, h);
                    img[h * bmp.Width + w] = 2.0f * pxl.R / 255.0f - 1;
                    if (chnl_cnt > 1)
                        img[bmp.Width * bmp.Height + h * bmp.Width + w] = 2.0f * pxl.G / 255.0f - 1;

                    if (chnl_cnt > 2)
                        img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] = 2.0f * pxl.B / 255.0f - 1;
                }

            bmp.Dispose();
            img_vec.Write(img);
            return img_vec;
        }
    }
}
