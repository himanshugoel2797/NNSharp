using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.MNIST
{
    public class Reader
    {
        const string Dataset = "I:/Datasets/mnist";
        const string TrainingImagesFile = "train-images-idx3-ubyte.gz";
        const string TrainingLabelsFile = "train-labels-idx1-ubyte.gz";
        const string TestImagesFile = "t10k-images-idx3-ubyte.gz";
        const string TestLabelsFile = "t10k-labels-idx1-ubyte.gz";

        private Matrix[] training_imgs, training_lbls;
        public Matrix[] TrainingImages { get => training_imgs; }
        public Matrix[] TrainingLabels { get => training_lbls; }


        public Reader()
        {

        }

        private int ReadInt32_BE(BinaryReader reader)
        {
            var bs = reader.ReadBytes(4);
            return BitConverter.ToInt32(bs.Reverse().ToArray(), 0);
        }

        public void InitializeTraining()
        {
            var training_img_p = Path.Combine(Dataset, TrainingImagesFile);
            var training_lbl_p = Path.Combine(Dataset, TrainingLabelsFile);

            List<Matrix> imgs = new List<Matrix>();
            List<Matrix> labels = new List<Matrix>();

            using (MemoryStream membuf = new MemoryStream(File.ReadAllBytes(training_img_p)))
            using (GZipStream gZipStream = new GZipStream(membuf, CompressionMode.Decompress))
            using (BinaryReader reader = new BinaryReader(gZipStream))
            {
                int v = ReadInt32_BE(reader);
                if (v != 2051)
                    throw new Exception();

                int cnt = ReadInt32_BE(reader);
                int rows = ReadInt32_BE(reader);
                int cols = ReadInt32_BE(reader);

                var m_f = new float[rows * cols];

                for (int i = 0; i < cnt; i++)
                {
                    Matrix m = new Matrix(rows * cols, 1, MemoryFlags.ReadWrite, false);
                    for (int y = 0; y < rows; y++)
                        for (int x = 0; x < cols; x++)
                            m_f[y * cols + x] = 2.0f * (reader.ReadByte()) / 255.0f - 1.0f;

                    m.Write(m_f);
                    imgs.Add(m);
                }
            }

            using (MemoryStream membuf = new MemoryStream(File.ReadAllBytes(training_lbl_p)))
            using (GZipStream gZipStream = new GZipStream(membuf, CompressionMode.Decompress))
            using (BinaryReader reader = new BinaryReader(gZipStream))
            {
                if (ReadInt32_BE(reader) != 2049)
                    throw new Exception();

                int cnt = ReadInt32_BE(reader);

                var m_f = new float[1];
                for (int i = 0; i < cnt; i++)
                {
                    Matrix m = new Matrix(1, 1, MemoryFlags.ReadWrite, false);
                    m_f[0] = reader.ReadByte();

                    m.Write(m_f);
                    labels.Add(m);
                }
            }

            training_imgs = imgs.ToArray();
            training_lbls = labels.ToArray();

            Random rng = new Random(0);
            for (int i = 0; i < training_imgs.Length; i++)
            {
                int swap_idx = rng.Next() % training_imgs.Length;
                var tmp_img = training_imgs[swap_idx];
                var tmp_lbl = training_lbls[swap_idx];

                training_imgs[swap_idx] = training_imgs[i];
                training_lbls[swap_idx] = training_lbls[i];

                training_imgs[i] = tmp_img;
                training_lbls[i] = tmp_lbl;
            }
        }



    }
}
