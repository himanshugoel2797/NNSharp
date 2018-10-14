using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test
{
    public class AnimeGAN
    {
        NeuralNetwork generator;
        NeuralNetwork discriminator;
        NeuralNetwork combined;

        SGD disable;

        const int Side = 16 * 4;
        const int InputSize = Side * Side * 3;
        const int LatentSize = 16 * 3;
        const int BatchSize = 32;

        const string TrainingDataPath = @"I:\Datasets\anime-faces\combined";
        const string TrainingDataPath_SMALL = @"I:\Datasets\anime-faces\combined\small";
        List<string> TrainingFiles;

        public AnimeGAN()
        {
            generator = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.00f))
                                .Optimizer(new SGD(), 0.3f)
                                .AddFC<ReLU>(LatentSize * 2)
                                .AddFC<ReLU>(LatentSize * 4)
                                .AddFC<Tanh>(LatentSize * 8)
                                .AddFC<Sigmoid>(InputSize)
                                .Build();

            discriminator = new NeuralNetworkBuilder(InputSize)
                                .LossFunction<GAN_DiscCrossEntropy>()
                                .WeightInitializer(new UniformWeightInitializer(1, 0.00f))
                                .Optimizer(new SGD(), 0.3f)
                                .AddFC<Sigmoid>(LatentSize * 4)
                                .AddFC<Sigmoid>(LatentSize)
                                .AddFC<Sigmoid>(16)
                                .AddFC<Sigmoid>(1)
                                .Build();

            combined = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                                .Optimizer(new SGD(), 0.03f / BatchSize)
                                .Add(generator)
                                .Add(discriminator)
                                .Build();

            disable = new SGD();
            disable.SetLearningRate(0);
        }

        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
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
                        bmp.Dispose();
                        rsz_bmp.Save(smallPath);
                        rsz_bmp.Dispose();
                    }

                    TrainingFiles.Add(smallPath);
                }
            }
        }

        private float[] LoadImage(string file)
        {
            var bmp = new Bitmap(file);
            float[] img = new float[InputSize];

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

            return img;
        }

        private void SaveImage(string file, float[] img)
        {
            var bmp = new Bitmap(Side, Side);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    //img[i] /= (2 * 1.7159f) + 0.5f;
                    //img[i + 1] /= (2 * 1.7159f) + 0.5f;
                    //img[i + 2] /= (2 * 1.7159f) + 0.5f;

                    if (img[i] < 0) img[i] = Math.Abs(img[i]);
                    if (img[i + 1] < 0) img[i + 1] = Math.Abs(img[i + 1]);
                    if (img[i + 2] < 0) img[i + 2] = Math.Abs(img[i + 2]);

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
                    img[i] -= img2[i];
                    img[i + 1] -= img2[i + 1];
                    img[i + 2] -= img2[i + 2];


                    //img[i] /= (2 * 1.7159f) + 0.5f;
                    //img[i + 1] /= (2 * 1.7159f) + 0.5f;
                    //img[i + 2] /= (2 * 1.7159f) + 0.5f;

                    if (img[i] < 0) img[i] = Math.Abs(img[i]);
                    if (img[i + 1] < 0) img[i + 1] = Math.Abs(img[i + 1]);
                    if (img[i + 2] < 0) img[i + 2] = Math.Abs(img[i + 2]);

                    if (img[i] > 1) img[i] = 1;
                    if (img[i + 1] > 1) img[i + 1] = 1;
                    if (img[i + 2] > 1) img[i + 2] = 1;


                    if (Math.Abs(img[i] - img2[i]) > 0.05) img[i] = 0;
                    if (Math.Abs(img[i + 1] - img2[i + 1]) > 0.05) img[i + 1] = 0;
                    if (Math.Abs(img[i + 2] - img2[i + 2]) > 0.05) img[i + 2] = 0;

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
            Vector data_test_vec = new Vector(InputSize, MemoryFlags.ReadOnly, false);
            Vector data_vec2 = new Vector(LatentSize, MemoryFlags.ReadOnly, false);

            Vector[] data_vec = new Vector[BatchSize];
            Vector[] expected_out = new Vector[BatchSize];
            for (int i = 0; i < data_vec.Length; i++)
            {
                data_vec[i] = new Vector(InputSize, MemoryFlags.ReadOnly, false);
                expected_out[i] = new Vector(1, MemoryFlags.ReadOnly, false);
            }

            var data_test = LoadImage(TrainingFiles[0]);
            data_test_vec.Write(data_test);

            float[] res = new float[InputSize];
            float[] output_data = new float[1];

            for (int i0 = 0; i0 < 100000; i0++)
            {
                {
                    var data = new float[LatentSize];
                    float sum = 0;
                    for (int j = 0; j < data.Length; j++)
                    {
                        data[j] = (float)(r2.NextDouble());
                        sum += data[j];
                    }
                    for (int j = 0; j < data.Length; j++)
                        data[j] *= sum;

                    data_vec2.Write(data);

                    var res_vec = generator.Forward(data_vec2);
                    res_vec.Read(res);
                    SaveImage($@"Data\DecResults\{i0}.png", res);

                    res_vec = discriminator.Forward(res_vec);
                    res_vec.Read(output_data);

                    Console.WriteLine($"SAVE [{i0}] GENERATOR, DISCRIMINATOR RATING {output_data[0]}");
                }

                {
                    var res_vec = discriminator.Forward(data_test_vec);
                    res_vec.Read(output_data);

                    Console.WriteLine($"[{i0}] TEST: {TrainingFiles[0]}, DISCRIMINATOR RATING {output_data[0]}");
                }

                for (int k = 0; k < 30; k++)
                {
                    for (int i = 0; i < data_vec.Length; i += 2)
                    {
                        {
                            int idx = (r.Next() % (TrainingFiles.Count / 2));
                            var data = LoadImage(TrainingFiles[idx]);
                            data_vec[i].Write(data);

                            output_data[0] = 1;
                            expected_out[i].Write(output_data);
                        }

                        {
                            var data = new float[LatentSize];
                            float sum = 0;
                            for (int j = 0; j < data.Length; j++)
                            {
                                data[j] = (float)(r2.NextDouble());
                                sum += data[j];
                            }
                            for (int j = 0; j < data.Length; j++)
                                data[j] *= sum;

                            data_vec2.Write(data);

                            var res_vec = generator.Forward(data_vec2);
                            res_vec.Read(res);
                            data_vec[i + 1].Write(res);

                            output_data[0] = 0;
                            expected_out[i + 1].Write(output_data);

                            List<IOptimizer> opts = new List<IOptimizer>();
                            for (int j = 0; j < discriminator.LayerCount; j++)
                            {
                                opts.Add(discriminator[j].GetOptimizer());
                                discriminator[j].SetOptimizer(disable);
                            }
                            discriminator.OutputError(data_vec[i + 1], expected_out[i + 1], out var delta, out var weights);
                            discriminator.OutputError(data_vec[i], expected_out[i], out var delta2, out var weights2);
                            for (int j = 0; j < discriminator.LayerCount; j++)
                            {
                                discriminator[j].SetOptimizer(opts[j]);
                            }

                            generator.Learn(data_vec2, delta, weights, out delta, out weights);
                            generator.Learn(data_vec2, delta2, weights2, out delta2, out weights2);

                            discriminator.TrainSingle(data_vec[i], expected_out[i]);
                            discriminator.TrainSingle(data_vec[i + 1], expected_out[i + 1]);
                        } 

                    }

                    //discriminator.TrainMultiple(data_vec, expected_out);

                    Console.Write($"{k},");
                }

                Console.WriteLine($"[{i0}]");
            }
            //encoder.Save($@"Data\encoder_final.bin");
            //decoder.Save($@"Data\decoder_final.bin");
            combined.Save($@"Data\combined_final.bin");
        }

    }
}
