using NNSharp3.ANN;
using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Drawing;
//using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SMath = System.Math;

namespace NNSharp3.Test
{
    public class AnimeAutoencoder
    {
        NeuralNetwork encoder;
        NeuralNetwork decoder;
        NeuralNetwork combined;

        const int Side = 32;
        const int InputSize = Side * Side * 3;
        const int LatentSize = 16 * 3;
        const int BatchSize = 64;

        const string TrainingDataPath = @"I:\Datasets\Lewds\Pictures";
        const string TrainingDataPath_SMALL = @"I:\Datasets\Lewds\Pictures\small";
        List<string> TrainingFiles;

        public AnimeAutoencoder()
        {
            encoder = new NeuralNetworkBuilder(InputSize)
                                .SetLossFunction(LossFunction.MeanSquaredError)
                                .SetWeightInitializer(WeightInitializer.UniformNoise, 0, 1, 0.0001f)
                                //.SetOptimizer()
                                .AddFCLayer(LatentSize, ActivationFunction.ReLU)
                                //.AddFCLayer(LatentSize, ActivationFunction.ReLU)
                                //.AddFCLayer(LatentSize, ActivationFunction.ReLU)
                                .Build();

            decoder = new NeuralNetworkBuilder(LatentSize)
                                .SetLossFunction(LossFunction.MeanSquaredError)
                                .SetWeightInitializer(WeightInitializer.UniformNoise, 0, 1, 0.0001f)
                                //.SetOptimizer()
                                .AddFCLayer(InputSize, ActivationFunction.Sigmoid)
                                .Build();

            //encoder = NeuralNetwork.Load($@"Data\encoder7000.bin");
            //decoder = NeuralNetwork.Load($@"Data\decoder7000.bin");


            /*SGD nSGD = new SGD();
            nSGD.SetLearningRate(0.00001f);
            for (int i = 0; i < encoder.LayerCount; i++)
                encoder.SetOptimizer(nSGD);

            for (int i = 0; i < decoder.LayerCount; i++)
                decoder.SetOptimizer(nSGD);
                */ 
            combined = new NeuralNetworkBuilder(InputSize)
                                .Add(encoder)
                                .Add(decoder)
                                .Build();
        }

        static Random rsz_rng = new Random(1);
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            float aspectRatio = (float)image.Width / image.Height;

            float height_var = (height - (int)(height * aspectRatio)) * 0.8f;
            float width_var = (width - (int)(width / aspectRatio)) * 0.8f;

            var destRect = new Rectangle((int)(height_var * rsz_rng.NextDouble()), 0, (int)(height * aspectRatio), width);
            if (aspectRatio > 1.0f)
                destRect = new Rectangle(0, (int)(width_var * rsz_rng.NextDouble()), height, (int)(width / aspectRatio));
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static string[] CutImage(Image image, int width, int height, string dst, string filename)
        {
            List<string> cut_paths = new List<string>();

            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            //destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            for (int y = 0; y < image.Height - height; y += height / 2)
                for (int x = 0; x < image.Width - width; x += width / 2)
                {
                    using (var graphics = Graphics.FromImage(destImage))
                    {
                        graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                        graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                        graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                        graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                        graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

                        using (var wrapMode = new ImageAttributes())
                        {
                            wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.Clamp);
                            //wrapMode.
                            graphics.DrawImage(image, destRect, x, y, width, height, GraphicsUnit.Pixel, wrapMode);
                        }
                    }

                    string path = Path.Combine(dst, filename + "_" + x + "_" + y + ".png");
                    cut_paths.Add(path);
                    destImage.Save(path);
                }

            destImage.Dispose();
            return cut_paths.ToArray();
        }

        public void InitializeDataset()
        {
            TrainingFiles = new List<string>();

            Directory.CreateDirectory(TrainingDataPath_SMALL);

            var files = Directory.EnumerateFiles(TrainingDataPath).ToArray();
            for (int i = 0; i < files.Length; i++)
            {
                if (new string[] { ".png", ".jpg" }.Contains(Path.GetExtension(files[i])))
                {
                    var smallPath = Path.Combine(TrainingDataPath_SMALL, Path.GetFileName(files[i]));

                    if (!File.Exists(smallPath))
                    {
                        var bmp = new Bitmap(files[i]);
                        var rsz_bmp = ResizeImage(bmp, Side, Side);
                        rsz_bmp.Save(smallPath);
                        rsz_bmp.Dispose();

                        //var cut_src = ResizeImage(bmp, Side * 2, (int)(Side * 2 * bmp.Height / (float)bmp.Width));
                        //TrainingFiles.AddRange(CutImage(cut_src, Side, Side, TrainingDataPath_SMALL, Path.GetFileNameWithoutExtension(files[i])));
                        bmp.Dispose();
                    }
                }
            }

            if (TrainingFiles.Count == 0)
            {
                Random rng = new Random();
                TrainingFiles.AddRange(Directory.EnumerateFiles(TrainingDataPath_SMALL));
                while (TrainingFiles.Count > 1000)
                {
                    TrainingFiles.RemoveAt(rng.Next() % TrainingFiles.Count);
                }
            }
            else
            {
                var img = new float[InputSize];
                for (int idx = 0; idx < TrainingFiles.Count; idx++)
                {
                    var bmp = new Bitmap(TrainingFiles[idx]);
                    bool isSame = true;

                    var pxl0 = bmp.GetPixel(0, 0);
                    byte pR = pxl0.R;
                    byte pG = pxl0.G;
                    byte pB = pxl0.B;

                    int i = 0;
                    for (int h = 0; h < bmp.Height; h++)
                        for (int w = 0; w < bmp.Width; w++)
                        {
                            var pxl = bmp.GetPixel(w, h);
                            if (pR != pxl.R | pG != pxl.G | pB != pxl.B)
                            {
                                isSame = false;
                                break;
                            }
                            pR = pxl.R;
                            pG = pxl.G;
                            pB = pxl.B;

                            img[i++] = pxl.R / 255.0f;
                            img[i++] = pxl.G / 255.0f;
                            img[i++] = pxl.B / 255.0f;
                        }

                    bmp.Dispose();

                    if (isSame)
                    {
                        File.Delete(TrainingFiles[idx]);
                        TrainingFiles.RemoveAt(idx);
                        idx--;
                    }
                }
            }
        }

        private void LoadImage(string file, float[] img)
        {
            var bmp = new Bitmap(file);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    var pxl = bmp.GetPixel(w, h);
                    img[i++] = pxl.R / 255.0f;
                    img[i++] = pxl.G / 255.0f;
                    img[i++] = pxl.B / 255.0f;
                }

            bmp.Dispose();
        }

        private void SaveImage(string file, float[] img)
        {
            var bmp = new Bitmap(Side, Side);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    if (img[i] < 0) img[i] = SMath.Abs(img[i]);
                    if (img[i + 1] < 0) img[i + 1] = SMath.Abs(img[i + 1]);
                    if (img[i + 2] < 0) img[i + 2] = SMath.Abs(img[i + 2]);

                    if (img[i] > 1) img[i] = 1;
                    if (img[i + 1] > 1) img[i + 1] = 1;
                    if (img[i + 2] > 1) img[i + 2] = 1;
                     
                    bmp.SetPixel(w, h, Color.FromArgb((int)(img[i++] * 255.0f), (int)(img[i++] * 255.0f), (int)(img[i++] * 255.0f)));
                }

            bmp.Save(file);
            bmp.Dispose();
        }

        private void SaveImageDiff(string file, float[] img, float[] img2)
        {
            var bmp = new Bitmap(Side, Side);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    if (img[i] < 0) img[i] = SMath.Abs(img[i]);
                    if (img[i + 1] < 0) img[i + 1] = SMath.Abs(img[i + 1]);
                    if (img[i + 2] < 0) img[i + 2] = SMath.Abs(img[i + 2]);

                    if (img[i] > 1) img[i] = 1;
                    if (img[i + 1] > 1) img[i + 1] = 1;
                    if (img[i + 2] > 1) img[i + 2] = 1;

                    bmp.SetPixel(w, h, Color.FromArgb((int)(img[i++] * 255.0f), (int)(img[i++] * 255.0f), (int)(img[i++] * 255.0f)));
                }

            bmp.Save(file);
            bmp.Dispose();
        }

        public void Train()
        {
            Directory.CreateDirectory(@"Data");
            Directory.CreateDirectory(@"Data\Results");
            Directory.CreateDirectory(@"Data\DecResults");
            Directory.CreateDirectory(@"Data\Diff");
            Directory.CreateDirectory(@"Data\DiffTest");


            Random r = new Random(0);
            Random r2 = new Random(0);


            float[] res1, res2, data;
            res1 = new float[InputSize];
            res2 = new float[InputSize];
            data = new float[LatentSize];

            Matrix data_vec = new Matrix(1, LatentSize);

            var dataset_vec = new Matrix[TrainingFiles.Count];
            float[][] dataset = new float[TrainingFiles.Count][];
            for (int i = 0; i < TrainingFiles.Count; i++)
            {
                dataset[i] = new float[InputSize];
                dataset_vec[i] = new Matrix(1, InputSize);
                LoadImage(TrainingFiles[i], dataset[i]);
                dataset_vec[i].Write(dataset[i]);
            }

            for (int i0 = 000; i0 < 25000; i0++)
            {
                if (i0 % 1000 == 0)
                {
                    //encoder.Save($@"Data\encoder{i0}.bin");
                    //decoder.Save($@"Data\decoder{i0}.bin");
                    //combined.Save($@"Data\combined{i0}.bin");
                }
                 
                //(var weights, var rows, var cols) = combined.State(0, NeuralNetwork.StateValue.Weights);
                  
                {
                    int idx = (r.Next() % (TrainingFiles.Count / 2));
                    var res_vec = combined.Forward(dataset_vec[idx]);
                    res_vec.Read(res1);
                    SaveImage($@"Data\DiffTest\{i0}.png", dataset[idx]);
                    SaveImage($@"Data\Results\{i0}.png", res1);

                    Console.WriteLine($"SAVE [{i0}] File: {Path.GetFileNameWithoutExtension(TrainingFiles[idx])}");
                }

                {
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)(-r2.NextDouble() * 1.5f);
                    data_vec.Write(data);

                    var res_vec = decoder.Forward(data_vec);
                    //res_vec = combined.Forward(res_vec);
                    res_vec.Read(res2); 
                    SaveImage($@"Data\DecResults\{i0}.png", res2);

                    Console.WriteLine($"SAVE [{i0}] DECODER");
                }


                for (int i = 0; i < BatchSize; i++)
                {
                    int idx = (r.Next() % TrainingFiles.Count / 2) + TrainingFiles.Count / 2;
                    combined.Forward(dataset_vec[idx]);
                    combined.Backward(dataset_vec[idx]);
                    combined.UpdateWeights(dataset_vec[idx], 0.001f);
                }

                Console.WriteLine($"[{i0}]");
            }
            //encoder.Save($@"Data\encoder_final2.bin");
            //decoder.Save($@"Data\decoder_final2.bin");
            //combined.Save($@"Data\combined_final2.bin");

            Console.WriteLine("DONE.");
        }

    }
}
