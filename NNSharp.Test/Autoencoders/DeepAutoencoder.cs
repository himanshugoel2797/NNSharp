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
    public class DeepAutoencoder : ITest
    {
        const int LatentSize = 64;
        const int Side = 96;
        const int ImageCount = 2000;
        const int Seed = 0;

        public void Run()
        {
            var inputDataset = new NoisyImageSet(@"I:\Datasets\anime-faces\combined", Side, ImageCount, Seed);
            inputDataset.Initialize();
            /*
            var encoder = new NeuralNetworkBuilder(Side * Side * 3)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddFC(128)
                                .AddActivation<LeakyReLU>()
                                .AddFC(100)
                                .AddActivation<LeakyReLU>()
                                .AddFC(80)
                                .AddActivation<LeakyReLU>()
                                .AddFC(70)
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .Build();

            var decoder = new NeuralNetworkBuilder(LatentSize)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddFC(70)
                                .AddActivation<LeakyReLU>()
                                .AddFC(80)
                                .AddActivation<LeakyReLU>()
                                .AddFC(100)
                                .AddActivation<LeakyReLU>()
                                .AddFC(128)
                                .AddActivation<LeakyReLU>()
                                .AddFC(Side * Side * 3)
                                .AddActivation<Tanh>()
                                .Build();
                                */
            var encoder = NeuralNetwork.Load(@"I:\NeuralNetworks\DeepAutoencoder_LowestError_Emilia_0\state_189270.enc");
            var decoder = NeuralNetwork.Load(@"I:\NeuralNetworks\DeepAutoencoder_LowestError_Emilia_0\state_189270.dec");

            var trainer = new AutoencoderTrainer("DeepAutoencoder Demo", encoder, decoder);
            trainer.SetDataset(inputDataset);

            LearningManager.Show(trainer);
        }
    }
}
