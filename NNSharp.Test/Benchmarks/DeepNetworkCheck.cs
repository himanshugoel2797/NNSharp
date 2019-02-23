using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.WeightInitializers;
using System;

namespace NNSharp.Test.Benchmarks
{
    public class DeepNetworkCheck : ITest
    {
        const int InputSize = 8;
        const int LatentSize = 16;

        public void Run()
        {
            BenchmarkHarness harness = new BenchmarkHarness("NeuralNet Check");
            var nn = new NeuralNetworkBuilder(InputSize)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.000f))
                                .AddFC(LatentSize * 4)
                                .AddActivation<LeakyReLU>()
                                .AddFC(LatentSize)
                                .AddActivation<LeakyReLU>()
                                .Build();

            for (int i = 0; i < 10; i++)
            {
                harness.Start();
                nn.Check(0.001f, i == 0);
                harness.Stop();
            }
            harness.Show();
        }
    }
}