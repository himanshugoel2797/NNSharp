using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Datasets;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkTrainer;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Autoencoders
{
    public class ConvAutoencoder : ITest
    {
        const int LatentSize = 2048;
        const int Side = 64;
        const int ImageCount = 34400;
        const int Seed = 0;

        public void Run()
        {
            var inputDataset = new UnlabeledImageSet(@"I:\Datasets\Gelbooru_SMALL", Side, ImageCount, Seed, false);

            var encoder = new NeuralNetworkBuilder(Side * Side * 3)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddConv(3, 3, 1, 0, Side, 3)
                                .AddActivation<LeakyReLU>()
                                .AddConv(3, 15, 1, 0, Side - 2, 3)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .Build();

            var decoder = new NeuralNetworkBuilder(LatentSize)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddFC((Side - 2) * (Side - 2) * 5)
                                .AddActivation<LeakyReLU>()
                                .AddConv(3, 3, 1, 2, (Side - 2), 5)
                                .AddActivation<Tanh>()
                                .Build();
            decoder.Check(10f, true);

            inputDataset.Initialize();
            var trainer = new AutoencoderTrainer("ConvAutoencoder Demo", encoder, decoder);
            trainer.SetDataset(inputDataset);

            LearningManager.Show(trainer);
        }
    }
}
