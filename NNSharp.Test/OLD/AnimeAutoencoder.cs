using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
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
    public class AnimeAutoencoder
    {
        NeuralNetwork encoder;
        NeuralNetwork decoder;
        NeuralNetwork combined;

        const int Side = 96;
        const int InputSize = Side * Side * 3;
        const int LatentSize = 128 * 3;
        const int BatchSize = 128;

        public AnimeAutoencoder()
        {
            encoder = new NeuralNetworkBuilder(InputSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(LatentSize * 4)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .Build();

            decoder = new NeuralNetworkBuilder(LatentSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize * 4)
                                .AddActivation<LeakyReLU>()
                                .AddFC(InputSize)
                                .AddActivation<Tanh>()
                                .Build();

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

            AnimeDatasets a_dataset = new AnimeDatasets(Side);
            a_dataset.InitializeDataset();

            SGD sgd = new SGD();
            sgd.SetLearningRate(0.000025f / BatchSize);

            NRandom r = new NRandom(0);
            NRandom r2 = new NRandom(0);

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

            for (int i0 = 000; i0 < 25000; i0++)
            {

                {
                    int idx = (r.Next() % (a_dataset.TrainingFiles.Count / 2));
                    var res_vec = combined.Forward(dataset_vec[idx]);
                    res_vec.Read(res1);
                    a_dataset.SaveImage($@"Data\DiffTest\{i0}.png", dataset[idx]);
                    a_dataset.SaveImage($@"Data\Results\{i0}.png", res1);

                    Console.WriteLine($"SAVE [{i0}] File: {Path.GetFileNameWithoutExtension(a_dataset.TrainingFiles[idx])}");
                }

                {
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)r2.NextGaussian(0, 1) / data.Length;
                    data_vec.Write(data);

                    var res_vec = decoder.Forward(data_vec);
                    res_vec.Read(res2);
                    a_dataset.SaveImage($@"Data\DecResults\{i0}.png", res2);

                    Console.WriteLine($"SAVE [{i0}] DECODER");
                }

                float err0 = 0;
                Vector[] batch = new Vector[BatchSize];
                for (int i = 0; i < BatchSize; i++)
                {
                    int idx = (r.Next() % a_dataset.TrainingFiles.Count / 2) + a_dataset.TrainingFiles.Count / 2;
                    batch[i] = dataset_vec[idx];
                }
                combined.TrainMultiple(batch, batch, sgd);

                err0 = combined.Error() / BatchSize;
                Console.WriteLine($"[{i0}] Error: {err0}");
                sgd.Update(err0);
                sgd.SetLearningRate(0.25f);// / BatchSize);
                if (err0 < err)
                {
                    encoder.Save($@"Data\encoder{i0}.bin");
                    decoder.Save($@"Data\decoder{i0}.bin");
                    combined.Save($@"Data\combined{i0}.bin");

                    err = err0;
                    if (err < 0.003f) break;
                }

            }
            encoder.Save($@"Data\encoder_final2.bin");
            decoder.Save($@"Data\decoder_final2.bin");
            combined.Save($@"Data\combined_final2.bin");

            Console.WriteLine("DONE.");
        }
    }
}
