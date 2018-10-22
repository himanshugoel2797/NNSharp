using NNSharp3.ANN;
using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.Test
{
    class Program
    {
        public static void PrintMatrix(float[] data, int rows, int cols)
        {
            string r = "";
            for (int col = 0; col < cols; col++)
            {
                r += "[";
                for (int row = 0; row < rows; row++)
                {
                    r += data[row * cols + col];
                    if (row < rows - 1) r += ", ";
                }

                r += "]\n";
            }
            Console.WriteLine(r);
        }

        static void Main(string[] args)
        {
            var dev = Device.GetDevice();
            dev.GLInfo();
            /*
            var nn = new NeuralNetworkBuilder(2)
                .SetWeightInitializer(WeightInitializer.UniformNoise, 0, 0.01f)
                .SetLossFunction(LossFunction.MeanSquaredError)
                .AddFCLayer(2, ActivationFunction.ReLU)
                //.AddFCLayer(1, ActivationFunction.Tanh)
                .Build();

            var inputs = new Matrix[4];
            var outputs = new Matrix[4];

            int i0 = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var i_d = new float[2];
                i_d[0] = (i & 1) * 0.5f;
                i_d[1] = ((i & 2) >> 1) * 0.5f;

                inputs[i0] = new Matrix(2, 1);
                inputs[i0].Write(i_d);

                var o_d = new float[2];
                o_d[0] = i_d[0];
                o_d[1] = i_d[1];
                outputs[i0] = new Matrix(2, 1);
                outputs[i0].Write(o_d);

                i0++;
            }

            for (int j = 0; j < 10000; j++)
                for (int i = 0; i < inputs.Length; i++)
                {
                    nn.Forward(inputs[i]);
                    nn.Backward(outputs[i]);
                    nn.UpdateWeights(inputs[i], 0.005f);
                }

            for (int i = 0; i < inputs.Length; i++)
            {
                var i_d = new float[2];
                inputs[i].Read(i_d);

                var o = nn.Forward(inputs[i]);

                var o_d = new float[2];
                o.Read(o_d);
                Console.WriteLine($"Inputs = [{i_d[0]}, {i_d[1]}], Output = [{o_d[0]},{ o_d[1]}], Expected = [{i_d[0] + i_d[1]}]");
            }

            for (int i = 0; i < nn.LayerCount; i++)
            {
                (var data, var rows, var cols) = nn.State(i, NeuralNetwork.StateValue.Weights);
                Console.WriteLine("\nWeights = ");
                PrintMatrix(data, rows, cols);

                (data, rows, cols) = nn.State(i, NeuralNetwork.StateValue.Biases);
                Console.WriteLine("Biases = ");
                PrintMatrix(data, rows, cols);

                (data, rows, cols) = nn.State(i, NeuralNetwork.StateValue.Errors);
                Console.WriteLine("Errors = ");
                PrintMatrix(data, rows, cols);
            }*/

            
            var ann = new AnimeAutoencoder();
            ann.InitializeDataset();
            ann.Train();

            Console.ReadLine();
        }
    }
}
