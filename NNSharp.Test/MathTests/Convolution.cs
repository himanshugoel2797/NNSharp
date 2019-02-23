using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.MathTests
{
    public class Convolution : ITest
    {
        class WeightInitializer : IWeightInitializer
        {
            Matrix w1;
            public WeightInitializer()
            {
                //w1 = new Matrix(2, 2, MemoryFlags.ReadWrite, false);
                //w1.Write(new float[] { 0.5f, 0.5f, 0.5f, 0.5f });
            }

            public float GetBias()
            {
                return 0;
            }

            public float GetWeight(int in_dim, int out_dim)
            {
                return 0.5f;
            }
        }

        public void Run()
        {

            Vector x1 = new Vector(9, MemoryFlags.ReadWrite, true);

            Vector x2 = new Vector(9, MemoryFlags.ReadWrite, true);
            x2.Write(new float[] { 1, 1, 1,
                                   0, 0, 0,
                                   0, 0, 0 });

            Vector x3 = new Vector(9, MemoryFlags.ReadWrite, true);
            x3.Write(new float[] { 0, 0, 0,
                                   1, 1, 1,
                                   1, 1, 1 });

            Vector x4 = new Vector(9, MemoryFlags.ReadWrite, true);
            x4.Write(new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 });

            Vector[] X = new Vector[] { x1, x2, x3, x4 };

            Vector[] Y = new Vector[4];
            for (int i = 0; i < Y.Length; i++) Y[i] = new Vector(1, MemoryFlags.ReadWrite, true);
            Y[0].Write(new float[] { 0.53f });
            Y[1].Write(new float[] { 0.77f });
            Y[2].Write(new float[] { 0.88f });
            Y[3].Write(new float[] { 1.1f });

            var sgd = new SGD();
            sgd.SetLearningRate(0.7f);

            var nn = new NeuralNetworkBuilder(9)
                .AddConv(2, 1, 1, 0, 3, 1)
                .AddActivation<Tanh>()
                .AddFC(1)
                .AddActivation<Sigmoid>()
                .LossFunction<Quadratic>()
                .WeightInitializer(new WeightInitializer())
                .Build();

            ConvLayer l0 = new ConvLayer();
            l0.SetFilterCount(1);
            l0.SetFilterSize(2);
            l0.SetInputDepth(1);
            l0.SetPaddingSize(0);
            l0.SetStrideLength(1);
            l0.SetInputSize(3);
            l0.SetWeights(new WeightInitializer());

            ActivationLayer aL0 = new ActivationLayer(new Tanh());
            aL0.SetInputSize(4);

            FCLayer l1 = new FCLayer(1);
            l1.SetInputSize(4);
            l1.SetWeights(new WeightInitializer());

            ActivationLayer aL1 = new ActivationLayer(new Sigmoid());
            aL1.SetInputSize(1);

            var res0 = l0.Forward(x2);
            var al0 = aL0.Forward(res0);
            var res1 = l1.Forward(al0);
            var al1 = aL1.Forward(res1);

            var data = al1.Read();

            for (int epoch = 0; epoch < 2; epoch++)
                for (int i = 0; i < X.Length; i++)
                {
                    //var res = nn.Forward(X[i]);
                    //nn.TrainMultiple(X, Y, sgd);
                    nn.TrainSingle(X[i], Y[i], sgd);
                    Console.WriteLine($"Current iter : {epoch} Current train: {i} Current cost: {nn.Error()}");
                    Console.WriteLine($"\ngrad_1 : { string.Join(", ", (nn.Layers[0] as ConvLayer).WeightErrors[0][0].Read()) }");
                    Console.WriteLine($"\ngrad_2 : { string.Join(", ", (nn.Layers[2] as FCLayer).WeightDelta.Read()) }");
                    Console.WriteLine($"\nlayer_1 : { string.Join(", ", (nn.Layers[1] as ActivationLayer).PrevInput.Read()) }");
                    Console.WriteLine($"\nlayer_2 : { string.Join(", ", (nn.Layers[3] as ActivationLayer).PrevInput.Read()) }");
                    Console.WriteLine($"\nw1 : { string.Join(", ", (nn.Layers[0] as ConvLayer).Weights[0][0].Read()) }");
                    //Console.WriteLine($"\nw2 : { string.Join(", ", (nn.Layers[2] as ConvLayer).Weights[0][0].Read()) }");
                    //Console.WriteLine($"layer_2_act : { string.Join(", ", (nn.Layers[3] as ActivationLayer).Activation.Read()) }");
                    Console.WriteLine("\n");
                }
        }
    }
}
