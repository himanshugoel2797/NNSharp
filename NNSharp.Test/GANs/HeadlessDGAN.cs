using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Datasets;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.GANs
{
    class HeadlessDGAN : ITest
    {
        const int LatentVectorLen = 128;
        const int Side = 32;
        const int ImageCount = 300;
        const int Seed = 0;

        public void Run()
        {
            //Load dataset
            var inputDataset = new UnlabeledImageSet(@"I:\Datasets\anime-faces\emilia_(re_zero)", Side, ImageCount, Seed, true);
            inputDataset.Initialize();

            //Create the generator
            var generator = new NeuralNetworkBuilder(LatentVectorLen)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddFC(256)
                                .AddActivation<Tanh>()
                                .AddFC(256)
                                .AddActivation<LeakyReLU>()
                                .AddFC(256)
                                .AddActivation<Tanh>()
                                .AddFC(512)
                                .AddActivation<LeakyReLU>()
                                .AddFC(2048)
                                .AddActivation<Tanh>()
                                .AddFC(Side * Side * 3)
                                .AddActivation<Sigmoid>()
                                .Build();

            //generator.Check(1f);

            //Create the discriminator
            var discriminator = new NeuralNetworkBuilder(Side * Side * 3)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddFC(512)
                                .AddActivation<LeakyReLU>()
                                .AddFC(256)
                                .AddActivation<LeakyReLU>()
                                .AddFC(32)
                                .AddActivation<LeakyReLU>()
                                .AddFC(1)
                                .AddActivation<Sigmoid>()
                                .Build();

            //var generator = NeuralNetwork.Load($@"Data\generator_final.bin");
            //var discriminator = NeuralNetwork.Load($@"Data\discriminator_final.bin");

            var gen_rng = new Vector(LatentVectorLen, MemoryFlags.ReadWrite, false);
            var gen_rng_data = new float[LatentVectorLen];
            var rng = new NRandom(0);

            var binary_crossentropy = new BinaryCrossEntropy();

            var d_real_loss = new Vector(1, MemoryFlags.ReadWrite, false);
            var d_fake_loss = new Vector(1, MemoryFlags.ReadWrite, false);
            var d_g_fake_loss = new Vector(1, MemoryFlags.ReadWrite, false);

            var real_const = new Vector(1, MemoryFlags.ReadWrite, false);
            real_const.Write(new float[] { 1 });

            var fake_const = new Vector(1, MemoryFlags.ReadWrite, false);
            fake_const.Write(new float[] { 0 });

            var tmp_d = new float[Side * Side * 3];
            var loss_d_tmp = new float[1];
            var loss_g_tmp = new float[1];

            var g_sgd = new SGD();
            g_sgd.SetLearningRate(0.0005f);

            var d_sgd = new SGD();
            d_sgd.SetLearningRate(0.0005f);

            int d_score = 0, g_score = 0;

            for (int idx = 0; idx < 80000000; idx++)
            {
                //Reset vectors
                Vector.Mult(d_real_loss, 0);
                Vector.Mult(d_fake_loss, 0);
                Vector.Mult(d_g_fake_loss, 0);

                //Get training data
                inputDataset.GetNextTrainingSet(out var input, out var dummy);

                //Generate gaussian random vector for generator
                for (int i = 0; i < gen_rng_data.Length; i++) gen_rng_data[i] = (float)rng.NextGaussian(0, 0.5f); /// gen_rng_data.Length;
                gen_rng.Write(gen_rng_data);


                //Forward prop discriminator with real item and calculate cross entropy loss
                var d_real_out = discriminator.Forward(input);
                d_real_out.Read(loss_d_tmp);
                binary_crossentropy.LossDeriv(d_real_out, real_const, d_real_loss);
                discriminator.Error(d_real_loss, true);

                //Forward prop discriminator with fake item and calculate cross entropy loss
                var gen_out = generator.Forward(gen_rng);
                var d_fake_out = discriminator.Forward(gen_out);
                d_fake_out.Read(loss_g_tmp);
                binary_crossentropy.LossDeriv(d_fake_out, fake_const, d_fake_loss);
                discriminator.Error(d_fake_loss, true);

                //Backward prop generator with the discriminator's fake cross entropy
                binary_crossentropy.LossDeriv(d_fake_out, real_const, d_g_fake_loss);
                var d_err = discriminator.Error(d_g_fake_loss, false);
                generator.TrainSingle(d_err, g_sgd);

                //Backward prop discriminator with average of real and fake cross entropy
                discriminator.Learn(d_sgd);
                discriminator.Reset();
                
                /*
                d_err.Read(tmp_d);
                for (int q = 0; q < tmp_d.Length; q++)
                    if (tmp_d[q] != 0)
                    {
                        Console.WriteLine("Gradient Found.");
                        break;
                    }
                    */

                if (loss_d_tmp[0] > loss_g_tmp[0])
                    d_score++;
                else
                    g_score++;

                if (idx % 100 == 0)
                {
                    Console.Clear();
                    Console.WriteLine($"Iteration [{idx}]: Real: {loss_d_tmp[0],4:0.0000}, Fake: {loss_g_tmp[0],4:0.0000}\nScores: \n\tG: {g_score:000000}, D: {d_score:000000}");
                    ImageManipulation.SaveImage($@"Data\Results\{idx / 100}.png", gen_out, Side);
                    generator.Save($@"Data\generator_final2.bin");
                    discriminator.Save($@"Data\discriminator_final2.bin");
                }
            }

            Console.ReadLine();
        }
    }
}
