using NNSharp;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkBuilder;
using NNSharp.ANN.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    public class GradientChecking
    {
        class ConstantWeightInitializer : IWeightInitializer
        {
            public float GetBias()
            {
                return 0;
            }

            public float GetWeight(int in_dim, int out_dim)
            {
                return 0.5f;
            }
        }

        LayerContainer front, back;

        public void Check()
        {
            front = InputLayer.Create(3, 1);
            back = ActivationLayer.Create<Sigmoid>();

            front.Append(
                ConvLayer.Create(2, 1).Append(
                ActivationLayer.Create<Tanh>().Append(
                FCLayer.Create(1, 1).Append(
                    back
            ))));

            front.SetupInternalState();
            front.InitializeWeights(new ConstantWeightInitializer());

            var lossFunc = new Quadratic();
            var optimizer = new SGD(0.7f);

            Matrix x0 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x0.Write(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 });

            Matrix x1 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x1.Write(new float[] { 1, 1, 1, 0, 0, 0, 0, 0, 0 });

            Matrix x2 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x2.Write(new float[] { 0, 0, 0, 1, 1, 1, 1, 1, 1 });

            Matrix x3 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x3.Write(new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 });



            var loss_vec = new Matrix(1, 1, MemoryFlags.ReadWrite, true);
            var x = new Matrix[] { x0, x1, x2, x3 };
            var y = new Matrix[]
            {
                new Matrix(1, 1, MemoryFlags.ReadWrite, true),
                new Matrix(1, 1, MemoryFlags.ReadWrite, true),
                new Matrix(1, 1, MemoryFlags.ReadWrite, true),
                new Matrix(1, 1, MemoryFlags.ReadWrite, true),
            };
            y[0].Write(new float[] { 0.53f });
            y[1].Write(new float[] { 0.77f });
            y[2].Write(new float[] { 0.88f });
            y[3].Write(new float[] { 1.1f });

            for (int epoch = 0; epoch < 200; epoch++)
                for (int idx = 0; idx < x.Length; idx++)
                {
                    var output = front.ForwardPropagate(x[idx]);

                    //Compute loss deriv
                    loss_vec.Clear();
                    lossFunc.LossDeriv(output[0], y[idx], loss_vec);

                    back.ResetLayerErrors();
                    back.ComputeGradients(loss_vec);
                    back.ComputeLayerErrors(loss_vec);
                    back.UpdateLayers(optimizer);
                }

        }
    }
}
