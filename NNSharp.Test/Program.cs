using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NNSharp.Test
{
    class Program
    {
        const int M = 16;
        const int K = 16;
        const int N = 16;

        const int TS = 16;

        static void Main(string[] args)
        {
            /*
            var in_sz = 4096;
            var out_sz = 16;

            var nn = new NeuralNetworkBuilder(in_sz)
                                .LossFunction<Quadratic>()
                                .WeightInitializer(new UniformWeightInitializer(0, 0.01f))
                                .Optimizer(new SGD(), 0.5f)
                                .AddFC<Sigmoid>(2048)
                                .AddFC<Sigmoid>(1024)
                                .AddFC<Sigmoid>(out_sz)
                                .Build();

            int cnt = 10;
            Vector[] inputs = new Vector[cnt];
            Vector[] outputs = new Vector[cnt];
            float[][] outputs_raw = new float[cnt][];

            Random rng = new Random(0);

            for (int i = 0; i < cnt; i++)
            {
                float[] i_s = new float[in_sz];
                float[] o_s = new float[out_sz];

                for (int j = 0; j < i_s.Length; j++)
                    i_s[j] = (float)rng.NextDouble();

                o_s[0] = i_s.Sum();

                inputs[i] = new Vector(in_sz, MemoryFlags.ReadOnly, false);
                inputs[i].Write(i_s);

                outputs[i] = new Vector(out_sz, MemoryFlags.ReadWrite, false);
                outputs[i].Write(o_s);
                outputs_raw[i] = o_s;
            }

            {
                int score = 0;
                for (int i = cnt / 2; i < cnt; i++)
                {
                    var res = nn.Forward(inputs[i]);
                    float[] res_data = new float[out_sz];
                    res.Read(res_data);
                    
                    Console.WriteLine($"{i}: RES:{res_data[0]}, EXP:{outputs_raw[i][0]}");

                    if (res_data[0] - outputs_raw[i][0] < 0.05f)
                        score++;

                }
                Console.WriteLine("Score: " + (float)score / cnt * 2);
            }

            for (int i0 = 0; i0 < 1000; i0++)
                for (int i = 0; i < cnt; i++)
                {
                    nn.TrainSingle(inputs[i], outputs[i]);
                }

            {
                int score = 0;
                for (int i = cnt / 2; i < cnt; i++)
                {
                    var res = nn.Forward(inputs[i]);
                    float[] res_data = new float[out_sz];
                    res.Read(res_data);

                    Console.WriteLine($"{i}: RES:{res_data[0]}, EXP:{outputs_raw[i][0]}");

                    if (res_data[0] - outputs_raw[i][0] < 0.05f)
                        score++;

                }
                Console.WriteLine("Score: " + (float)score / cnt * 2);
            }*/

            Vector a = new Vector(10, MemoryFlags.ReadWrite, false);
            float[] a_data = new float[a.Length];
            Vector b = new Vector(10, MemoryFlags.ReadWrite, false);
            float[] b_data = new float[b.Length];

            Vector res = new Vector(10, MemoryFlags.ReadWrite, true);
            a.Write(a_data);

            b.Write(b_data);


            /*while (true)
            {
                Vector.Hadamard(a, b, res);
            }*/

            var autoencoder = new AnimeAutoencoder();
            autoencoder.InitializeDataset();
            autoencoder.Train();

            Console.ReadLine();
        }
    }
}
