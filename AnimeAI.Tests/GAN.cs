using NNSharp;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkBuilder;
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
    class GAN
    {
        LayerContainer discriminator, discriminator_back, generator, generator_back;

        const int StartSide = 256;
        const int LatentSize = 22 * 22 * 8;
        const int EndSide = 256;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 16;

        public GAN()
        {
            var pooling_0 = PoolingLayer.Create(2, 2);
            var pooling_1 = PoolingLayer.Create(2, 2);
            var pooling_2 = PoolingLayer.Create(2, 2);
            var pooling_3 = PoolingLayer.Create(2, 2);

            discriminator = InputLayer.Create(StartSide, 3);
            discriminator_back = ActivationLayer.Create<Sigmoid>();
            discriminator.Append(
                ConvLayer.Create(3, 8).Append(              //o = 254
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 252
                ActivationLayer.Create<LeakyReLU>().Append(
                pooling_0.Append(                           //o = 126
                ConvLayer.Create(3, 8).Append(              //o = 124
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 122
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 120
                ActivationLayer.Create<LeakyReLU>().Append(
                pooling_1.Append(                           //o = 60
                ConvLayer.Create(3, 8).Append(              //o = 58
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 56
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 54
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 52
                ActivationLayer.Create<LeakyReLU>().Append(
                pooling_2.Append(                           //o = 26
                ConvLayer.Create(3, 8).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8).Append(              //o = 22
                ActivationLayer.Create<LeakyReLU>().Append(
                pooling_3.Append(                           //o = 11
                FCLayer.Create(1, 1).Append(
                    discriminator_back
            ))))))))))))))))))))))))))));

            generator = InputLayer.Create(22, 8);
            generator_back = ActivationLayer.Create<Tanh>();
            generator.Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                UnpoolingLayer.Create(pooling_2).Append(       //o = 52
                ConvLayer.Create(3, 8, 2).Append(              //o = 54
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 56
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 58
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 60
                ActivationLayer.Create<LeakyReLU>().Append(
                UnpoolingLayer.Create(pooling_1).Append(       //o = 120
                ConvLayer.Create(3, 8, 2).Append(              //o = 122
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 124
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 126
                ActivationLayer.Create<LeakyReLU>().Append(
                UnpoolingLayer.Create(pooling_0).Append(       //o = 252
                ConvLayer.Create(3, 8, 2).Append(              //o = 254
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 3, 2).Append(              //o = 256
                    generator_back
            )))))))))))))))))))))))));

            //Initialize Weights
            discriminator.SetupInternalState();
            discriminator.InitializeWeights(new UniformWeightInitializer(0, 0.001f));

            generator.SetupInternalState();
            generator.InitializeWeights(new UniformWeightInitializer(0, 0.001f));
        }

        public void Train()
        {
            string dir = "GAN_Data";

            Directory.CreateDirectory($@"{dir}");
            Directory.CreateDirectory($@"{dir}\Results");
            Directory.CreateDirectory($@"{dir}\Sources");

            #region GAN Variables
            var sgd_disc = new Adam(0.001f, 1e-6f);
            var sgd_gen = new Adam(0.001f, 1e-6f);
            var bce = new BinaryCrossEntropy();

            NRandom r_dataset = new NRandom(0);
            NRandom r_latent = new NRandom(0);

            Matrix data_vec = new Matrix(LatentSize, 1, MemoryFlags.ReadOnly, false);
            Matrix d_real_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix d_fake_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix g_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix loss_reader = new Matrix(1, 1, MemoryFlags.ReadWrite, true);

            float d_real_loss_f = 0;
            float d_fake_loss_f = 0;
            float g_loss_f = 0;

            Matrix zero = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            zero.Memory[0] = 0;

            Matrix one = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            one.Memory[0] = 1;
            #endregion


            #region Setup Database
            AnimeDatasets dataset = new AnimeDatasets(StartSide, @"I:\Datasets\VAE_Dataset\White", @"I:\Datasets\VAE_Dataset\White\conv");
            dataset.InitializeDataset();
            Matrix[] dataset_vec = new Matrix[dataset.TrainingFiles.Count];
            float[][] dataset_f = new float[dataset.TrainingFiles.Count][];

            for (int i = 0; i < dataset.TrainingFiles.Count; i++)
            {
                dataset_f[i] = new float[InputSize];
                dataset_vec[i] = new Matrix(InputSize, 1, MemoryFlags.ReadOnly, false);
                dataset.LoadImage(dataset.TrainingFiles[i], dataset_f[i]);
                dataset_vec[i].Write(dataset_f[i]);
            }
            #endregion

            for (int i0 = 000; i0 < 5000 * BatchSize; i0++)
            {
                int idx = (r_dataset.Next() % dataset.TrainingFiles.Count);

                //Generate the fake data
                for (int i1 = 0; i1 < LatentSize; i1++)
                    data_vec.Memory[i1] = (float)r_latent.NextGaussian(0, 1);
                var fake_result = generator.ForwardPropagate(data_vec);

                //Reset layer errors
                generator_back.ResetLayerErrors();
                discriminator_back.ResetLayerErrors();

                d_fake_loss.Clear();
                d_real_loss.Clear();
                g_loss.Clear();

                //Discriminator feed forward for real data
                var d_real_class = discriminator.ForwardPropagate(dataset_vec[idx]);
                bce.LossDeriv(d_real_class[0], one, d_real_loss);
                var d_real_prop = discriminator_back.ComputeGradients(d_real_loss);
                discriminator_back.ComputeLayerErrors(d_real_loss);
                bce.Loss(d_real_class[0], one, loss_reader);
                d_real_loss_f = loss_reader.Memory[0];
                loss_reader.Memory[0] = 0;

                //Discriminator feed forward for fake data
                var d_fake_class = discriminator.ForwardPropagate(fake_result);
                bce.LossDeriv(d_fake_class[0], zero, d_fake_loss);
                var d_fake_prop = discriminator_back.ComputeGradients(d_fake_loss);
                discriminator_back.ComputeLayerErrors(d_fake_loss);
                bce.Loss(d_fake_class[0], zero, loss_reader);
                d_fake_loss_f = loss_reader.Memory[0];
                loss_reader.Memory[0] = 0;

                //Compute discriminator crossentropy loss assuming fake is real and propagate
                bce.LossDeriv(d_fake_class[0], one, g_loss);
                var d_err = discriminator_back.ComputeGradients(g_loss);
                generator_back.ComputeGradients(d_err);
                generator_back.ComputeLayerErrors(d_err);
                bce.Loss(d_fake_class[0], one, loss_reader);
                g_loss_f = loss_reader.Memory[0];
                loss_reader.Memory[0] = 0;

                //Update layers
                generator_back.UpdateLayers(sgd_gen);
                discriminator_back.UpdateLayers(sgd_disc);

                if (i0 % BatchSize == 0)
                {
                    dataset.SaveImage($@"{dir}\Sources\{i0 / BatchSize}.png", dataset_f[idx]);
                    dataset.SaveImage($@"{dir}\Results\{i0 / BatchSize}.png", fake_result[0].Read());
                }

                Console.Clear();
                Console.WriteLine($"Iteration: {i0 / BatchSize} Sub-batch: {i0 % BatchSize}");
                Console.WriteLine($"Discriminator Real Loss: {d_real_loss_f}\nDiscriminator Fake Loss: {d_fake_loss_f}\nGenerator Loss: {g_loss_f}");
            }

            discriminator.Save($@"{dir}\network_final.bin");
            Console.WriteLine("DONE.");
        }
    }
}
