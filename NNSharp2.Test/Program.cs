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
            Vector input = new Vector("input", 1, 2048);
            Vector expectedOutput = new Vector("expectedOutput", 1, 2048);

            int layerCount = 2;
            var weights = new Matrix[layerCount];
            var biases = new Vector[layerCount];

            var shdw_weights = new Matrix[layerCount];
            var shdw_biases = new Vector[layerCount];

            var grad = new Matrix[layerCount];

            for (int i = 0; i < layerCount; i++)
            {
                if (i == 0)
                {
                    weights[i] = new Matrix("weights_" + i, 2048, 256);
                    biases[i] = new Vector("biases_" + i, 1, 256);
                }
                else if (i == 1)
                {
                    weights[i] = new Matrix("weights_" + i, 256, 2048);
                    biases[i] = new Vector("biases_" + i, 1, 2048);
                }
            }

            var intermediate = weights[0] * input + biases[0];
            var intermediate_activ = Vector.Tanh(intermediate);

            var output = weights[1] * intermediate_activ + biases[1];
            var output_activ = Vector.Tanh(output);

            var loss = 0.5f * Vector.Power((expectedOutput - output_activ), 2);

            //loss.Gradient(weights[0]);
            //Console.WriteLine(loss.Gradient(weights[0]).Item2);

            Device.Add("loss_deriv_wrt_output_activ", loss.Gradient(output_activ).Item2);
            Device.Add("output_activ_deriv_wrt_output", output_activ.Gradient(output).Item2);
            Device.Add("output_deriv_wrt_intermediate_activ", output.Gradient(intermediate_activ).Item2);
            Device.Add("intermediate_deriv_wrt_weights_0", intermediate.Gradient(weights[0]).Item2);



            //grad[1] = (Matrix)Device.Add("grad_1", (Matrix)loss.Gradient(weights[1]).Item2);
            //grad[0] = (Matrix)Device.Add("grad_0", (Matrix)loss.Gradient(weights[0]).Item2);

            //shdw_weights[1] = (Matrix)Device.Add("shdw_weights_1", weights[1] - 0.005f * grad[1]);
            //shdw_weights[0] = (Matrix)Device.Add("shdw_weights_0", weights[0] - 0.005f * grad[0]);

            //Device.AddRoot(output_activ);
            //Device.AddRoot(loss.Gradient(weights[1]));
            //Device.AddRoot(Matrix.Hadamard(Matrix.Hadamard(loss.Gradient(output_activ), output_activ.Gradient(output)), output.Gradient(weights[1])));

            //Device.AddRoot(loss.Gradient(weights[0]));

            //var tmp = Matrix.Hadamard(loss.Gradient(intermediate_activ), intermediate_activ.Gradient(intermediate));
            //Device.AddRoot(Matrix.Hadamard(tmp, intermediate.Gradient(weights[0])));


            //Device.AddRoot(weights[1]);
            //Device.AddRoot(weights[0]);
            Device.Build();
        }
    }
}
