using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkBuilder;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.MNIST
{
    public class DGAN
    {
        LayerContainer discriminator, discriminator_back, generator, generator_back;

        const int StartSide = 28;
        const int LatentSize = 100;
        const int EndSide = 28;
        const int InputSize = StartSide * StartSide;
        const int OutputSize = EndSide * EndSide;

        const int BatchSize = 16;

        public DGAN()
        {
            discriminator = InputLayer.Create(StartSide, 1);
            discriminator_back = ActivationLayer.Create<Sigmoid>();
            discriminator.Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(1, 1).Append(
                    discriminator_back
            ))));

            generator = InputLayer.Create(1, LatentSize);
            generator_back = ActivationLayer.Create<Tanh>();
            generator.Append(
                FCLayer.Create(1, 256).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.5f).Append(
                FCLayer.Create(1, 512).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                    DropoutLayer.Create(0.5f).Append(
                FCLayer.Create(1, OutputSize).Append(
                    generator_back
            ))))))));

            //Initialize Weights
            discriminator.SetupInternalState();
            discriminator.InitializeWeights(new UniformWeightInitializer(3, 0));

            generator.SetupInternalState();
            generator.InitializeWeights(new UniformWeightInitializer(1, 0));
        }

        public void Train()
        {
            string dir = "GAN_Data";

            Directory.CreateDirectory($@"{dir}");
            Directory.CreateDirectory($@"{dir}\Results");
            Directory.CreateDirectory($@"{dir}\Sources");

            #region GAN Variables
            var sgd_disc = new Adam(0.0002f, 1e-6f);
            var sgd_gen = new Adam(0.0002f, 1e-6f);

            var fake_loss = new NamedLossFunction(NamedLossFunction.GANDiscFake, NamedLossFunction.GANDiscFake);
            var real_loss = new NamedLossFunction(NamedLossFunction.GANDiscReal, NamedLossFunction.GANDiscReal);
            var gen_loss = new NamedLossFunction(NamedLossFunction.GANGen, NamedLossFunction.GANGen);

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
            float d_real_class_f = 0, d_fake_class_f = 0, g_class_f = 0;

            Matrix zero = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            zero.Memory[0] = 0;

            Matrix one = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            one.Memory[0] = 1;
            #endregion


            #region Setup Database
            Reader dataset = new Reader();
            dataset.InitializeTraining();
            Matrix[] imgs = dataset.TrainingImages;
            #endregion

            for (int i0 = 000; i0 < 1000 * BatchSize; i0++)
            {
                int idx = 0;
                idx = (r_dataset.Next() % imgs.Length);

                //Generate the fake data
                for (int i1 = 0; i1 < LatentSize; i1++)
                    data_vec.Memory[i1] = (float)r_latent.NextGaussian(0, 1);//LatentSize;
                var fake_result = generator.ForwardPropagate(data_vec);

                if (i0 % BatchSize == 0)
                {
                    SaveImage($@"{dir}\Sources\{i0 / BatchSize}.png", imgs[idx].Read(), 28);
                    SaveImage($@"{dir}\Results\{i0 / BatchSize}.png", fake_result[0].Read(), 28);
                }

                Console.Clear();
                Console.WriteLine($"Iteration: {i0 / BatchSize} Sub-batch: {i0 % BatchSize}");
                Console.WriteLine($"Discriminator Real Loss: {d_real_loss_f}\nDiscriminator Fake Loss: {d_fake_loss_f}\nGenerator Loss: {g_loss_f}\n");
                Console.WriteLine($"Discriminator Real Prediction: {d_real_class_f}\nDiscriminator Fake Prediction: {d_fake_class_f}\nGenerator Prediction: {g_class_f}");

                d_fake_loss.Clear();
                d_real_loss.Clear();
                g_loss.Clear();

                zero.Memory[0] = (r_latent.Next() % 100) / 1000f;
                one.Memory[0] = 1 - (r_latent.Next() % 100) / 1000f;

                //Discriminator feed forward for real data
                {
                    var d_real_class = discriminator.ForwardPropagate(imgs[idx]);
                    real_loss.LossDeriv(d_real_class[0], one, d_real_loss, 0.01f * sgd_disc.L2Val / sgd_disc.Net);

                    var d_real_prop = discriminator_back.ComputeGradients(d_real_loss);
                    discriminator_back.ComputeLayerErrors(d_real_loss);

                    d_real_class_f = d_real_class[0].Memory[0];
                    real_loss.Loss(d_real_class[0], one, loss_reader, 0.01f * sgd_disc.L2Val);
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

        public void SaveImage(string file, float[] img, int Side)
        {
            var bmp = new Bitmap(Side, Side);

            try
            {
                for (int h = 0; h < bmp.Height; h++)
                    for (int w = 0; w < bmp.Width; w++)
                        bmp.SetPixel(w, h, Color.FromArgb((int)((img[h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f)));
            }
            catch (Exception) { }

            bmp.Save(file);
            bmp.Dispose();
        }

    }
}
