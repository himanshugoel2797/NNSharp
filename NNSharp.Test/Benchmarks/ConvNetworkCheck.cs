using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.WeightInitializers;
using System;

namespace NNSharp.Test.Benchmarks
{
    public class ConvNetworkCheck : ITest
    {
        const int InputSize = 1;
        const int LatentSize = 3;

        public void Run()
        {
            BenchmarkHarness harness = new BenchmarkHarness("NeuralNet Check");
            var nn = new NeuralNetworkBuilder(1)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(5 * 5 * 3)
                                //.AddActivation<LeakyReLU>()
                                .AddConv(3, 1, 1, 0, 5, 3)
                                //.AddActivation<LeakyReLU>()
                                .Add(new PoolingLayer(2, 2, 1))
                                //.AddFC(1)
                                //.AddActivation<LeakyReLU>()
                                .Build();

            for (int i = 0; i < 10; i++)
            {
                harness.Start();
                nn.Check(0.1f, i == 0);
                harness.Stop();
            }
            harness.Show();
        }
    }
}