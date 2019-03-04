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
        LayerContainer discriminator, discriminator_back, generator, generator_back, encoder, encoder_back;

        const int StartSide = 32;
        const int LatentSize = 100;
        const int EndSide = 32;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 16;

        public GAN()
        {

            #region 128x128
            /*discriminator = InputLayer.Create(StartSide, 3);
            discriminator_back = ActivationLayer.Create<Sigmoid>();
            discriminator.Append(
                ConvLayer.Create(3, 8).Append(              //o = 126
                ActivationLayer.Create<LeakyReLU>().Append(
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
                ConvLayer.Create(3, 8).Append(              //o = 22
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(1, 1).Append(
                    discriminator_back
            )))))))))))))))))))))))))));

            generator = InputLayer.Create(32, 8);
            generator_back = ActivationLayer.Create<Tanh>();
            generator.Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 3, 2).Append(              //o = 26
                    generator_back
            ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));*/
            #endregion

            /*
            discriminator = InputLayer.Create(StartSide, 3);
            discriminator_back = ActivationLayer.Create<Sigmoid>();
            discriminator.Append(
                ConvLayer.Create(3, 8).Append(              //o = 30
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 512).Append(              //o = 28
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 256).Append(              //o = 26
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 256).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 128).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 64).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 32).Append(              //o = 24
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.2f).Append(
                FCLayer.Create(1, 1).Append(
                    discriminator_back
            )))))))))))))))))))))));

            generator = InputLayer.Create(1, LatentSize);
            generator_back = ActivationLayer.Create<Tanh>();
            generator.Append(
                FCLayer.Create(1, 32).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 64).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 128).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 256).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 256).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 512).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(16, 2).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                ConvLayer.Create(3, 8, 2).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(EndSide, 3).Append(              //o = 18
                ActivationLayer.Create<LeakyReLU>().Append(
                    generator_back
            ))))))))))))))))))))))))));*/


            discriminator = InputLayer.Create(StartSide, 3);
            discriminator_back = ActivationLayer.Create<Sigmoid>();
            discriminator.Append(
                FCLayer.Create(1, 1024).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 512).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 64).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 64).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.3f).Append(
                FCLayer.Create(1, 1).Append(
                    discriminator_back
            ))))))))))))))))))));

            generator = InputLayer.Create(1, LatentSize);
            generator_back = ActivationLayer.Create<Tanh>();
            generator.Append(
                FCLayer.Create(1, 128).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                    DropoutLayer.Create(0.5f).Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 512).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 1024).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(32, 3).Append(
                    generator_back
            )))))))))))))));

            encoder = InputLayer.Create(32, 3);
            encoder_back = ActivationLayer.Create<LeakyReLU>();
            encoder.Append(
                FCLayer.Create(1, 1024).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 512).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                    DropoutLayer.Create(0.5f).Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, 128).Append(
                ActivationLayer.Create<ReLU>().Append(
                FCLayer.Create(1, LatentSize).Append(
                    encoder_back
            )))))))))))))));
            
            //Initialize Weights
            discriminator.SetupInternalState();
            discriminator.InitializeWeights(new UniformWeightInitializer(3, 0));

            generator.SetupInternalState();
            generator.InitializeWeights(new UniformWeightInitializer(1, 0));

            encoder.SetupInternalState();
            encoder.InitializeWeights(new UniformWeightInitializer(2, 0));
        }

        public void Train()
        {
            string dir = "GAN_Data";

            Directory.CreateDirectory($@"{dir}");
            Directory.CreateDirectory($@"{dir}\Results");
            Directory.CreateDirectory($@"{dir}\Sources");
            Directory.CreateDirectory($@"{dir}\ResultsPRE");
            Directory.CreateDirectory($@"{dir}\SourcesPRE");

            #region GAN Variables
            var sgd_disc = new Adam(0.0002f, 1e-6f);
            var sgd_gen = new Adam(0.0002f, 1e-6f);
            var sgd_dec = new Adam(0.0002f);

            var fake_loss = new NamedLossFunction(NamedLossFunction.GANDiscFake, NamedLossFunction.GANDiscFake);
            var real_loss = new NamedLossFunction(NamedLossFunction.GANDiscReal, NamedLossFunction.GANDiscReal);
            var gen_loss = new NamedLossFunction(NamedLossFunction.GANGen, NamedLossFunction.GANGen);
            var enc_loss = new Quadratic();

            NRandom r_dataset = new NRandom(0);
            NRandom r_latent = new NRandom(0);

            Matrix data_vec = new Matrix(LatentSize, 1, MemoryFlags.ReadOnly, false);
            Matrix d_real_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix d_fake_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix g_loss = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            Matrix e_loss = new Matrix(OutputSize, 1, MemoryFlags.ReadWrite, true);
            Matrix loss_reader = new Matrix(1, 1, MemoryFlags.ReadWrite, true);

            float d_real_loss_f = 0;
            float d_fake_loss_f = 0;
            float g_loss_f = 0;
            float d_real_class_f = 0, d_fake_class_f = 0, g_class_f = 0;

            Matrix zero = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            zero.Memory[0] = 0;

            Matrix one = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            one.Memory[0] = 1;
            #endregion


            #region Setup Database
            AnimeDatasets dataset = new AnimeDatasets(StartSide, /*@"I:\Datasets\anime-faces\combined", @"I:\Datasets\anime-faces\combined_small");*/@"I:\Datasets\VAE_Dataset\White", @"I:\Datasets\VAE_Dataset\White\conv");
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

            //Pretrain generator
            for (int i0 = 0; i0 < 5000; i0++)
            {
                int idx = 0;
                idx = (r_dataset.Next() % dataset.TrainingFiles.Count);

                var latent = encoder.ForwardPropagate(dataset_vec[idx]);
                var res = generator.ForwardPropagate(latent);

                e_loss.Clear();
                enc_loss.LossDeriv(res[0], dataset_vec[idx], e_loss, 0.0f * sgd_dec.L2Val / sgd_dec.Net);

                var enc_loss_v = generator_back.ComputeGradients(e_loss);
                generator_back.ComputeLayerErrors(e_loss);
                encoder_back.ComputeGradients(enc_loss_v);
                encoder_back.ComputeLayerErrors(enc_loss_v);

                sgd_dec.Update(0);
                generator_back.UpdateLayers(sgd_gen);
                encoder_back.UpdateLayers(sgd_dec);

                if (i0 % BatchSize == 0)
                {
                    dataset.SaveImage($@"{dir}\SourcesPRE\{i0 / BatchSize}.png", dataset_f[idx]);
                    dataset.SaveImage($@"{dir}\ResultsPRE\{i0 / BatchSize}.png", res[0].Read());
                    generator.Save($@"{dir}\pretrained_generator_fc.bin");
                    encoder.Save($@"{dir}\trained_encoder_fc.bin");
                }

                Console.Clear();
                Console.WriteLine($"Iteration: {i0 / BatchSize} Sub-batch: {i0 % BatchSize}");
            }

            for (int i0 = 000; i0 < 5000 * BatchSize; i0++)
            {
                int idx = 0;
                idx = (r_dataset.Next() % dataset.TrainingFiles.Count);

                //Generate the fake data
                for (int i1 = 0; i1 < LatentSize; i1++)
                    data_vec.Memory[i1] = (float)r_latent.NextGaussian(0, 1);//LatentSize;
                var fake_result = generator.ForwardPropagate(data_vec);

                if (i0 % BatchSize == 0)
                {
                    dataset.SaveImage($@"{dir}\Sources\{i0 / BatchSize}.png", dataset_f[idx]);
                    dataset.SaveImage($@"{dir}\Results\{i0 / BatchSize}.png", fake_result[0].Read());
                }

                Console.Clear();
                Console.WriteLine($"Iteration: {i0 / BatchSize} Sub-batch: {i0 % BatchSize}");
                Console.WriteLine($"Discriminator Real Loss: {d_real_loss_f}\nDiscriminator Fake Loss: {d_fake_loss_f}\nGenerator Loss: {g_loss_f}\n");
                Console.WriteLine($"Discriminator Real Prediction: {d_real_class_f}\nDiscriminator Fake Prediction: {d_fake_class_f}\nGenerator Prediction: {g_class_f}");

                d_fake_loss.Clear();
                d_real_loss.Clear();
                g_loss.Clear();

                zero.Memory[0] = (r_latent.Next() % 1000) / 10000f;
                one.Memory[0] = 1 - (r_latent.Next() % 1000) / 10000f;

                //Discriminator feed forward for real data
                {
                    var d_real_class = discriminator.ForwardPropagate(dataset_vec[idx]);
                    real_loss.LossDeriv(d_real_class[0], one, d_real_loss, 0.01f * sgd_disc.L2Val / sgd_disc.Net);

                    var d_real_prop = discriminator_back.ComputeGradients(d_real_loss);
                    discriminator_back.ComputeLayerErrors(d_real_loss);

                    d_real_class_f = d_real_class[0].Memory[0];
                    real_loss.Loss(d_real_class[0], one, loss_reader, 0.01f * sgd_disc.L2Val / sgd_disc.Net);
                    d_real_loss_f = loss_reader.Memory[0];
                    loss_reader.Memory[0] = 0;
                }

                //Discriminator feed forward for fake data
                {
                    var d_fake_class = discriminator.ForwardPropagate(fake_result);
                    fake_loss.LossDeriv(d_fake_class[0], zero, d_fake_loss, 0.01f * sgd_disc.L2Val / sgd_disc.Net);

                    var d_fake_prop = discriminator_back.ComputeGradients(d_fake_loss);
                    discriminator_back.ComputeLayerErrors(d_fake_loss);

                    d_fake_class_f = d_fake_class[0].Memory[0];
                    fake_loss.Loss(d_fake_class[0], zero, loss_reader, 0.01f * sgd_disc.L2Val / sgd_disc.Net);
                    d_fake_loss_f = loss_reader.Memory[0];
                    loss_reader.Memory[0] = 0;
                }

                //Update and reset discriminator
                sgd_disc.Update(0);
                discriminator_back.UpdateLayers(sgd_disc);
                discriminator_back.ResetLayerErrors();

                //Generate the fake data again
                {
                    for (int i1 = 0; i1 < LatentSize; i1++)
                        data_vec.Memory[i1] = (float)r_latent.NextGaussian(0, 1);//LatentSize;
                    fake_result = generator.ForwardPropagate(data_vec);
                    var d_gen_class = discriminator.ForwardPropagate(fake_result);

                    //Compute discriminator crossentropy loss assuming fake is real and propagate
                    gen_loss.LossDeriv(d_gen_class[0], one, g_loss, 0.01f * sgd_gen.L2Val / sgd_gen.Net);
                    var d_err = discriminator_back.ComputeGradients(g_loss);
                    generator_back.ComputeGradients(d_err);
                    generator_back.ComputeLayerErrors(d_err);

                    g_class_f = d_gen_class[0].Memory[0];
                    gen_loss.Loss(d_gen_class[0], one, loss_reader, 0.01f * sgd_gen.L2Val / sgd_gen.Net);
                    g_loss_f = loss_reader.Memory[0];
                    loss_reader.Memory[0] = 0;

                    //Update generator
                    sgd_gen.Update(0);
                    generator_back.UpdateLayers(sgd_gen);
                    generator_back.ResetLayerErrors();
                    discriminator.ResetLayerErrors();
                }

            }

            discriminator.Save($@"{dir}\network_final.bin");
            Console.WriteLine("DONE.");
        }
    }
}
