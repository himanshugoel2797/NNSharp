using Accord.Imaging;
using Accord.Imaging.Filters;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Datasets;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkTrainer;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Classifiers
{
    class TextClassifier : ITest
    {
        const int Side = 32;

        char[] letters = new char[]
        {
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '(', ')', '+', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        };
        string[] lettersStr;

        int i1 = 0;
        private IDataset GetDataset()
        {
            for (int s = 10; s < 20; s++)
                for (int i = 0; i < letters.Length; i++)
                {
                    {
                        Bitmap tmp = new Bitmap(32, 32);
                        using (var g = Graphics.FromImage(tmp))
                        {
                            g.FillRectangle(Brushes.Black, 0, 0, 32, 32);
                            g.DrawString(letters[i].ToString(), new Font("Euro Caps", s), Brushes.White, 0, 0);
                        }

                        var img = Grayscale.CommonAlgorithms.BT709.Apply(tmp);
                        OtsuThreshold threshold = new OtsuThreshold();
                        threshold.ApplyInPlace(img);

                        var blob_ext = new BlobCounter(img);
                        blob_ext.FilterBlobs = true;
                        var blobs = blob_ext.GetObjects(img, false);
                        blobs = blobs.OrderBy((a) => a.Rectangle.X).ToArray();
                        for (int i0 = 0; i0 < blobs.Length; i0++)
                            using (var letter = blobs[i0].Image.ToManagedImage())
                            {
                                //Resize each character to 20x20
                                var l0 = new Bitmap(32, 32);
                                using (var g = Graphics.FromImage(l0))
                                {
                                    g.FillRectangle(Brushes.Black, 0, 0, l0.Width, l0.Height);

                                    int w = (int)(20 * (float)letter.Width / letter.Height);
                                    int h = 20;

                                    g.DrawImage(letter, (l0.Width - w) / 2, (l0.Height - h) / 2, w, h);
                                }

                                l0.Save($"Data2/{i1++}.png");
                                l0.Dispose();
                            }
                    }
                }

            var dataset = new LabeledFileImageSet("Data2", 32, 1, letters.Length, 1, 0);

            for (int i = 0; i < i1; i++)
            {
                var tag = new float[letters.Length];
                tag[i % letters.Length] = 1;
                dataset.AddFile($"Data2/{i}.png", tag);
            }

            dataset.Initialize();
            return dataset;
        }

        public void Run()
        {
            lettersStr = new string[letters.Length];
            for (int i = 0; i < letters.Length; i++) lettersStr[i] = letters[i].ToString();

            var inputDataset = GetDataset();

            var classifier = new NeuralNetworkBuilder(Side * Side)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                //.AddFC(512)
                                //.AddActivation<ReLU>()
                                .AddFC(256)
                                .AddActivation<LeakyReLU>()
                                .AddFC(128)
                                .AddActivation<LeakyReLU>()
                                .AddFC(64)
                                .AddActivation<LeakyReLU>()
                                .AddFC(letters.Length)
                                .AddActivation<Sigmoid>()
                                .Build();

            var trainer = new ClassifierTrainer("Text Classifier", lettersStr, classifier);
            trainer.SetDataset(inputDataset);

            LearningManager.Show(trainer);
        }
    }
}
