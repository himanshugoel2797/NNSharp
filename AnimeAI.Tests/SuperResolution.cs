using NNSharp;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    class SuperResolution
    {
        NeuralNetwork superres_enc, superres_dec, superres_comb;

        const int StartSide = 32;
        const int LatentSize = 256;
        const int EndSide = 64;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 64;

        public SuperResolution()
        {
            superres_enc = new NeuralNetworkBuilder(InputSize)
                .LossFunction<Quadratic>()
                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                .AddFC(512)
                .AddActivation<LeakyReLU>()
                .AddFC(256)
                .AddActivation<LeakyReLU>()
                .AddFC(LatentSize)
                .AddActivation<LeakyReLU>()
                .Build();

            superres_dec = new NeuralNetworkBuilder(LatentSize)
                .LossFunction<Quadratic>()
                .WeightInitializer(new UniformWeightInitializer(0, 0.001f))
                .AddFC(256)
                .AddActivation<LeakyReLU>()
                .AddFC(256)
                .AddActivation<LeakyReLU>()
                .AddFC(256)
                .AddActivation<LeakyReLU>()
                .AddFC(512)
                .AddActivation<LeakyReLU>()
                .AddFC(512)
                .AddActivation<LeakyReLU>()
                .AddFC(1024)
                .AddActivation<LeakyReLU>()
                .AddFC(OutputSize)
                .AddActivation<Sigmoid>()
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
            Directory.CreateDirectory(@"DAutoencoder_Data");
            Directory.CreateDirectory(@"DAutoencoder_Data\Results");
            Directory.CreateDirectory(@"DAutoencoder_Data\DecResults");
            Directory.CreateDirectory(@"DAutoencoder_Data\Diff");
            Directory.CreateDirectory(@"DAutoencoder_Data\DiffTest");
            Directory.CreateDirectory(@"DAutoencoder_Data\Filters");

            AnimeDatasets a_dataset = new AnimeDatasets(StartSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_small");
            a_dataset.InitializeDataset();

            AnimeDatasets b_dataset = new AnimeDatasets(EndSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_large");
            b_dataset.InitializeDataset();

            SGD sgd = new SGD();
            sgd.SetLearningRate(0.05f);

            Random r = new Random(0);
            Random r2 = new Random(0);

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
                    a_dataset.SaveImage($@"DAutoencoder_Data\DiffTest\{i0}.png", a_dataset_f[idx]);
                    b_dataset.SaveImage($@"DAutoencoder_Data\Results\{i0}.png", res1);

                    Console.WriteLine($"SAVE [{i0}] File: {Path.GetFileNameWithoutExtension(a_dataset.TrainingFiles[idx])}");
                }

                {
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)(2 * r2.NextDouble() - 1);
                    data_vec.Write(data);

                    var res_vec = superres_dec.Forward(data_vec);
                    res_vec.Read(res2);
                    b_dataset.SaveImage($@"DAutoencoder_Data\DecResults\{i0}.png", res2);

                    Console.WriteLine($"SAVE [{i0}] DECODER");
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
                sgd.SetLearningRate(0.025f);// * (float)Math.Exp(-i0 * 0.001f));
                if (err0 < err)
                {
                    if (err0 < 0.01f)
                    {
                        superres_enc.Save($@"DAutoencoder_Data\encoder{i0}.bin");
                        superres_dec.Save($@"DAutoencoder_Data\decoder{i0}.bin");
                        superres_comb.Save($@"DAutoencoder_Data\combined{i0}.bin");
                    }

                    err = err0;
                }

            }
            superres_enc.Save($@"DAutoencoder_Data\encoder_f.bin");
            superres_dec.Save($@"DAutoencoder_Data\decoder_f.bin");
            superres_comb.Save($@"DAutoencoder_Data\combined_f.bin");

            Console.WriteLine("DONE.");
        }
    }
}
