using NNSharp;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    class ConvSuperResolution
    {
        NeuralNetwork superres_enc, superres_dec, superres_comb;

        const int StartSide = 32;
        const int LatentSize = 2 * 2 * 8;
        const int EndSide = 32;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 64;

        public ConvSuperResolution()
        {
            superres_enc = new NeuralNetworkBuilder(InputSize)
                .LossFunction<Quadratic>()
                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                
                .AddConv(3, 16, 1, 0, StartSide, 3)
                .AddActivation<Tanh>()
                .AddPooling(2, 2, 16)
                .AddConv(3, 8, 1, 2, (StartSide - 2)/2, 16)
                .AddActivation<Tanh>()
                .AddPooling(2, 2, 8)
                .AddConv(3, 8, 1, 0, 8, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 0, 6, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 0, 4, 8)
                .AddActivation<Tanh>()
                .Build();
            /*
            superres_dec = new NeuralNetworkBuilder(LatentSize)
                .LossFunction<Quadratic>()
                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                .AddFC(128)
                .AddActivation<Tanh>()
                .AddFC(128)
                .AddActivation<Tanh>()
                .AddFC(128)
                .AddActivation<Tanh>()
                .AddFC(OutputSize)
                .AddActivation<Tanh>()
                .Build();
                */
            superres_dec = new NeuralNetworkBuilder(LatentSize)
                .LossFunction<Quadratic>()
                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                .AddConv(3, 8, 1, 2, 2, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 2, 4, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 2, 6, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 2, 8, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 8, 1, 2, 10, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 12, 8)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 14, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 16, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 18, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 20, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 22, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 24, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 26, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 6, 1, 2, 28, 6)
                .AddActivation<Tanh>()
                .AddConv(3, 3, 1, 2, 30, 6)
                .AddActivation<Tanh>()
                .Build();


            superres_comb = new NeuralNetworkBuilder(InputSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                                .Add(superres_enc)
                                .Add(superres_dec)
                                .Build();
        }

        public void Train()
        {
            Directory.CreateDirectory(@"DConvAutoencoder_Data");
            Directory.CreateDirectory(@"DConvAutoencoder_Data\Results");
            Directory.CreateDirectory(@"DConvAutoencoder_Data\DecResults");
            Directory.CreateDirectory(@"DConvAutoencoder_Data\Diff");
            Directory.CreateDirectory(@"DConvAutoencoder_Data\DiffTest");
            Directory.CreateDirectory(@"DConvAutoencoder_Data\Filters");

            AnimeDatasets a_dataset = new AnimeDatasets(StartSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_small");
            a_dataset.InitializeDataset();

            AnimeDatasets b_dataset = new AnimeDatasets(EndSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_small");
            b_dataset.InitializeDataset();

            SGD sgd = new SGD();
            sgd.SetLearningRate(0.05f);

            NRandom r = new NRandom(0);
            NRandom r2 = new NRandom(0);

            float[] res1, res2, data;
            res1 = new float[OutputSize];
            res2 = new float[OutputSize];
            data = new float[LatentSize];
            float err = float.MaxValue;

            Vector data_vec = new Vector(LatentSize, MemoryFlags.ReadOnly, false);

            Vector[] a_dataset_vec = new Vector[a_dataset.TrainingFiles.Count];
            float[][] a_dataset_f = new float[a_dataset.TrainingFiles.Count][];

            Vector[] b_dataset_vec = new Vector[a_dataset.TrainingFiles.Count];
            float[][] b_dataset_f = new float[a_dataset.TrainingFiles.Count][];

            for (int i = 0; i < a_dataset.TrainingFiles.Count; i++)
            {
                a_dataset_f[i] = new float[InputSize];
                a_dataset_vec[i] = new Vector(InputSize, MemoryFlags.ReadOnly, false);
                a_dataset.LoadImage(a_dataset.TrainingFiles[i], a_dataset_f[i]);
                a_dataset_vec[i].Write(a_dataset_f[i]);

                b_dataset_f[i] = new float[OutputSize];
                b_dataset_vec[i] = new Vector(OutputSize, MemoryFlags.ReadOnly, false);
                b_dataset.LoadImage(b_dataset.TrainingFiles[i], b_dataset_f[i]);
                b_dataset_vec[i].Write(b_dataset_f[i]);
            }

            for (int i0 = 000; i0 < (1 << 30) / BatchSize; i0++)
            {

                int idx = (r.Next() % (a_dataset.TrainingFiles.Count / 2));
                {
                    var res_vec = superres_comb.Forward(a_dataset_vec[idx]);
                    res_vec.Read(res1);
                    a_dataset.SaveImage($@"DConvAutoencoder_Data\DiffTest\{i0}.png", a_dataset_f[idx]);
                    b_dataset.SaveImage($@"DConvAutoencoder_Data\Results\{i0}.png", res1);

                    Console.WriteLine($"SAVE [{i0}] File: {Path.GetFileNameWithoutExtension(a_dataset.TrainingFiles[idx])}");
                }

                {
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)(r2.NextGaussian(0, 0.05f));
                    data_vec.Write(data);

                    var res_vec = superres_dec.Forward(data_vec);
                    res_vec.Read(res2);
                    b_dataset.SaveImage($@"DConvAutoencoder_Data\DecResults\{i0}.png", res2);

                    Console.WriteLine($"SAVE [{i0}] DECODER");
                }

                {
                    var conv_out = superres_enc.Layers[1].Forward(superres_enc.Layers[0].Forward(a_dataset_vec[idx]));
                    float[] conv_out_d = new float[conv_out.Length];
                    conv_out.Read(conv_out_d);

                    var out_side = (superres_enc.Layers[0] as ConvLayer).GetFlatOutputSize();
                    for (int i = 0; i < conv_out_d.Length / (out_side * out_side); i++)
                    {
                        a_dataset.SaveImage($@"DConvAutoencoder_Data\Filters\filter_{i}.png", conv_out_d, i * out_side * out_side, out_side);
                    }
                }

                float err0 = 0;
                for (int i = 0; i < BatchSize; i++)
                {
                    int b_idx = (r.Next() % a_dataset.TrainingFiles.Count);// + a_dataset.TrainingFiles.Count / 2;
                    superres_comb.TrainSingle(a_dataset_vec[b_idx], b_dataset_vec[b_idx], sgd);
                    err0 += superres_comb.Error();

                    if (i % 10 == 0)
                        Console.Write(i / 10 + ",");
                }

                err0 /= BatchSize;
                Console.WriteLine($"[{i0}] Error: {err0}");
                sgd.Update(err0);
                sgd.SetLearningRate(0.5f);// * (float)Math.Exp(-i0 * 0.001f));
                if (err0 < err)
                {
                    if (err0 < 0.01f)
                    {
                        superres_enc.Save($@"DConvAutoencoder_Data\encoder{i0}.bin");
                        superres_dec.Save($@"DConvAutoencoder_Data\decoder{i0}.bin");
                        superres_comb.Save($@"DConvAutoencoder_Data\combined{i0}.bin");
                    }

                    err = err0;
                }

            }
            superres_enc.Save($@"DConvAutoencoder_Data\encoder_f.bin");
            superres_dec.Save($@"DConvAutoencoder_Data\decoder_f.bin");
            superres_comb.Save($@"DConvAutoencoder_Data\combined_f.bin");

            Console.WriteLine("DONE.");
        }
    }
}
