using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
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

namespace NNSharp.Test.OLD
{
    public class ConvolutionalAutoencoder
    {
        NeuralNetwork encoder;
        NeuralNetwork decoder;
        NeuralNetwork combined;

        const int Side = 32;
        const int InputSize = Side * Side * 3;
        const int LatentSize = 8;
        const int BatchSize = 32;

        public ConvolutionalAutoencoder()
        {
            /*encoder = new NeuralNetworkBuilder(Side * Side * 3)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddConv(3, 3, 1, 0, Side, 3)
                                .AddActivation<Tanh>()
                                .Add(new PoolingLayer(2, 2, 3))
                                .AddConv(3, 10)
                                .AddActivation<LeakyReLU>()
                                .Add(new PoolingLayer(2, 2, 10))
                                .AddFC(LatentSize)
                                .AddActivation<Tanh>()
                                .Build();

            decoder = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(LatentSize)
                                .AddFC(32 * 32) 
                                .AddActivation<Tanh>()
                                //.AddConv(3, 3, 3, 128, 32, 10)
                                .AddFC(Side * Side * 3)
                                .AddActivation<Sigmoid>()
                                .Build();*/

            /*encoder = new NeuralNetworkBuilder(Side * Side * 3)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(32 * 8)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32 * 2)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .Build();

            decoder = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(32) 
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32 * 2)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32 * 8)
                                .AddActivation<LeakyReLU>()
                                //.AddConv(3, 3, 3, 128, 32, 10)
                                .AddFC(Side * Side * 3)
                                .AddActivation<Sigmoid>()
                                .Build();*/

            encoder = new NeuralNetworkBuilder(Side * Side * 3)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddConv(3, 5, 1, 0, Side, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 5, 1, 0, 30, 5)
                                .AddActivation<Tanh>()
                                .AddConv(3, 5, 1, 0, 28, 5)
                                .AddActivation<Tanh>()
                                .AddConv(3, 5, 1, 0, 26, 5)
                                .AddActivation<Tanh>()
                                .AddConv(3, 5, 1, 0, 24, 5)
                                .AddActivation<Tanh>()
                                .AddPooling(2, 2, 5)
                                .AddFC(LatentSize)
                                .AddActivation<Tanh>()
                                .Build();

            decoder = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(64)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 8, 1)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 10, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 12, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 14, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 16, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 18, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 20, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 22, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 24, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 26, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 28, 3)
                                .AddActivation<Tanh>()
                                .AddConv(3, 3, 1, 2, 30, 3)
                                .AddActivation<Sigmoid>()
                                .Build();

            //encoder = NeuralNetwork.Load(@"Data\encoder55761.bin");
            //decoder = NeuralNetwork.Load(@"Data\decoder55761.bin");

            combined = new NeuralNetworkBuilder(InputSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.00f))
                                .Add(encoder)
                                .Add(decoder)
                                .Build();

            
        }

        public void Train()
        {
            Directory.CreateDirectory(@"Data");
            Directory.CreateDirectory(@"Data\Results");
            Directory.CreateDirectory(@"Data\DecResults");
            Directory.CreateDirectory(@"Data\Diff");
            Directory.CreateDirectory(@"Data\DiffTest");
            Directory.CreateDirectory(@"Data\Filters");

            AnimeDatasets a_dataset = new AnimeDatasets(Side);
            a_dataset.InitializeDataset();

            SGD sgd = new SGD();
            sgd.SetLearningRate(0.005f);

            Random r = new Random(0);
            Random r2 = new Random(0);

            float[] res1, res2, data;
            res1 = new float[InputSize];
            res2 = new float[InputSize];
            data = new float[LatentSize];
            float err = float.MaxValue;

            Vector data_vec = new Vector(LatentSize, MemoryFlags.ReadOnly, false);

            Vector[] dataset_vec = new Vector[a_dataset.TrainingFiles.Count];
            float[][] dataset = new float[a_dataset.TrainingFiles.Count][];
            for (int i = 0; i < a_dataset.TrainingFiles.Count; i++)
            {
                dataset[i] = new float[InputSize];
                dataset_vec[i] = new Vector(InputSize, MemoryFlags.ReadOnly, false);
                a_dataset.LoadImage(a_dataset.TrainingFiles[i], dataset[i]);
                dataset_vec[i].Write(dataset[i]);
            }

            for (int i0 = 000; i0 < (1 << 30) / BatchSize; i0++)
            {

                int idx = (r.Next() % (a_dataset.TrainingFiles.Count / 2));
                {
                    var res_vec = combined.Forward(dataset_vec[idx]);
                    res_vec.Read(res1);
                    a_dataset.SaveImage($@"Data\DiffTest\{i0}.png", dataset[idx]);
                    a_dataset.SaveImage($@"Data\Results\{i0}.png", res1);

                    Console.WriteLine($"SAVE [{i0}] File: {Path.GetFileNameWithoutExtension(a_dataset.TrainingFiles[idx])}");
                }

                {
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)(2 * r2.NextDouble() - 1);
                    data_vec.Write(data);

                    var res_vec = decoder.Forward(data_vec);
                    res_vec.Read(res2);
                    a_dataset.SaveImage($@"Data\DecResults\{i0}.png", res2);

                    Console.WriteLine($"SAVE [{i0}] DECODER");
                }

                {
                    var conv_out = encoder.Layers[1].Forward(encoder.Layers[0].Forward(dataset_vec[idx]));
                    float[] conv_out_d = new float[conv_out.Length];
                    conv_out.Read(conv_out_d);

                    var out_side = (encoder.Layers[0] as ConvLayer).GetFlatOutputSize();
                    for (int i = 0; i < conv_out_d.Length / (out_side * out_side); i++)
                    {
                        a_dataset.SaveImage($@"Data\Filters\filter_{i}.png", conv_out_d, i * out_side * out_side, out_side);
                    }
                }

                float err0 = 0;
                for (int i = 0; i < BatchSize; i++)
                {
                    int b_idx = (r.Next() % a_dataset.TrainingFiles.Count);// + a_dataset.TrainingFiles.Count / 2;
                    combined.TrainSingle(dataset_vec[b_idx], dataset_vec[b_idx], sgd);
                    err0 += combined.Error();

                    if (i % 1000 == 0)
                        Console.Write(i / 1000 + ",");
                }

                err0 /= BatchSize;
                Console.WriteLine($"[{i0}] Error: {err0}"); 
                sgd.Update(err0);
                sgd.SetLearningRate(0.5f);// * (float)Math.Exp(-i0 * 0.001f));
                if (err0 < err)
                {
                    if (err0 < 0.01f)
                    {
                        encoder.Save($@"Data\encoder{i0}.bin");
                        decoder.Save($@"Data\decoder{i0}.bin");
                        combined.Save($@"Data\combined{i0}.bin");
                    }

                    err = err0;
                }

            }
            encoder.Save($@"Data\encoder_final2.bin");
            decoder.Save($@"Data\decoder_final2.bin");
            combined.Save($@"Data\combined_final2.bin");

            Console.WriteLine("DONE.");
        }
    }
}
