using NNSharp2.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix input = new Matrix("input", 1, 2048);
            Matrix expectedOutput = new Matrix("expectedOutput", 1, 2048);

            int layerCount = 2;
            Matrix[] weights = new Matrix[layerCount];
            Matrix[] biases = new Matrix[layerCount];

            for (int i = 0; i < layerCount; i++)
            {
                if (i == 0)
                {
                    weights[i] = new Matrix("weights_" + i, 2048, 256);
                    biases[i] = new Matrix("biases_" + i, 1, 256);
                }
                else if (i == 1)
                {
                    weights[i] = new Matrix("weights_" + i, 256, 2048);
                    biases[i] = new Matrix("biases_" + i, 1, 2048);
                }
            }

            Matrix intermediate = weights[0] * input + biases[0];
            Matrix intermediate_activ = Matrix.Tanh(intermediate);

            Matrix output = weights[1] * intermediate_activ + biases[1];
            Matrix output_activ = Matrix.Tanh(output);

            Matrix loss = 0.5f * Matrix.Power((expectedOutput - output_activ), 2);

            //weights[1] -= Matrix.Hadamard(loss.Gradient(weights[1]) * intermediate.Transpose(), new Matrix(256, 2048, 0.005f));
            //weights[0] -= Matrix.Hadamard(loss.Gradient(weights[0]) * input.Transpose(), new Matrix(2048, 256, 0.005f));

            //Device.AddRoot(output_activ);
            //Device.AddRoot(loss.Gradient(weights[1]));
            //Device.AddRoot(Matrix.Hadamard(Matrix.Hadamard(loss.Gradient(output_activ), output_activ.Gradient(output)), output.Gradient(weights[1])));

            Device.AddRoot(loss.Gradient(weights[0]));

            var tmp = Matrix.Hadamard(loss.Gradient(intermediate_activ), intermediate_activ.Gradient(intermediate));
            Device.AddRoot(Matrix.Hadamard(tmp, intermediate.Gradient(weights[0])));


            //Device.AddRoot(weights[1]);
            //Device.AddRoot(weights[0]);
            Device.Build();
        }
    }
}
