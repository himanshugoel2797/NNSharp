using NNSharp;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.NetworkBuilder;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
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

        LayerContainer front, back, conv, fc;

        public void Check()
        {
            front = InputLayer.Create(3, 1);
            back = ActivationLayer.Create<Sigmoid>();
            conv = ConvLayer.Create(2, 1);
            fc = FCLayer.Create(1, 1);

            front.Append(
                conv.Append(
                    ConvLayer.Create(3, 3, 2).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                fc.Append(
                FCLayer.Create(1, 1).Append(
                    back
            ))))));

            front.SetupInternalState();
            front.InitializeWeights(new UniformWeightInitializer(0, 0)); //ConstantWeightInitializer());

            var lossFunc = new Quadratic();
            var optimizer = new SGD(0.7f);

            Matrix x0 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x0.Write(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 });

            Matrix x1 = new Matrix(9, 1, MemoryFlags.ReadWrite, false);
            x1.Write(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f });

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

            float delta = 1e-1f;
            float orig_loss_deriv = 0;
            float norm_conv = 0.0f, norm_conv_net = 0.0f;
            float norm_fc = 0.0f, norm_fc_net = 0.0f;

            for (int epoch = 0; epoch < 1; epoch++)
                for (int idx = 1; idx < x.Length - 2; idx++)
                {
                    {
                        var output = front.ForwardPropagate(x[idx]);

                        //Compute loss deriv
                        loss_vec.Clear();
                        lossFunc.LossDeriv(output[0], y[idx], loss_vec, 0);
                        orig_loss_deriv = loss_vec.Memory[0];

                        back.ComputeGradients(loss_vec);
                        back.ComputeLayerErrors(loss_vec);
                    }

                    {
                        //Save weights and apply deltas
                        var conv_l = conv.CurrentLayer as ConvLayer;

                        for (int f_i = 0; f_i < conv_l.FilterCnt; f_i++)
                            for (int i_i = 0; i_i < conv_l.InputDepth; i_i++)
                                for (int f_y = 0; f_y < conv_l.FilterSz; f_y++)
                                    for (int f_x = 0; f_x < conv_l.FilterSz; f_x++)
                                    {
                                        var w_delta = conv_l.WeightErrors[f_i][i_i].Memory[f_y * conv_l.FilterSz + f_x];

                                        conv_l.Weights[f_i][i_i].Memory[f_y * conv_l.FilterSz + f_x] += delta;
                                        var output = front.ForwardPropagate(x[idx]);
                                        loss_vec.Clear();
                                        lossFunc.Loss(output[0], y[idx], loss_vec, 0);
                                        var y1 = loss_vec.Memory[0];

                                        conv_l.Weights[f_i][i_i].Memory[f_y * conv_l.FilterSz + f_x] -= 2 * delta;
                                        output = front.ForwardPropagate(x[idx]);
                                        loss_vec.Clear();
                                        lossFunc.Loss(output[0], y[idx], loss_vec, 0);
                                        var y0 = loss_vec.Memory[0];

                                        conv_l.Weights[f_i][i_i].Memory[f_y * conv_l.FilterSz + f_x] += delta;

                                        var deriv = (y1 - y0) / (2 * delta);
                                        var norm = ((w_delta - deriv) * (w_delta - deriv)) / ((w_delta + deriv) * (w_delta + deriv));
                                        norm_conv += norm;
                                        norm_conv_net++;
                                    }

                        var fc_l = fc.CurrentLayer as FCLayer;

                        for (int i = 0; i < fc_l.Weights.Rows * fc_l.Weights.Columns; i++)
                        {
                            var w_delta = fc_l.WeightDelta.Memory[i];

                            fc_l.Weights.Memory[i] += delta;
                            var output = front.ForwardPropagate(x[idx]);
                            loss_vec.Clear();
                            lossFunc.Loss(output[0], y[idx], loss_vec, 0);
                            var y1 = loss_vec.Memory[0];

                            fc_l.Weights.Memory[i] -= 2 * delta;
                            output = front.ForwardPropagate(x[idx]);
                            loss_vec.Clear();
                            lossFunc.Loss(output[0], y[idx], loss_vec, 0);
                            var y0 = loss_vec.Memory[0];

                            fc_l.Weights.Memory[i] += delta;

                            var deriv = (y1 - y0) / (2 * delta);
                            var norm = ((w_delta - deriv) * (w_delta - deriv)) / ((w_delta + deriv) * (w_delta + deriv));
                            norm_fc += norm;
                            norm_fc_net++;
                        }

                    }

                    {
                        back.ResetLayerErrors();
                        back.ComputeGradients(loss_vec);
                        back.ComputeLayerErrors(loss_vec);
                        back.UpdateLayers(optimizer);
                    }
                }

            Console.WriteLine($"Conv Norm {norm_conv / norm_conv_net}");
            Console.WriteLine($"FC Norm {norm_fc / norm_fc_net}");
            Console.ReadLine();
        }
    }
}
