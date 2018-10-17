using NNSharp3.AGNN;
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
        static void Main(string[] args)
        {
            //var dev = Device.GetDevice();
            //dev.GLInfo();
            /*
            var nn = new NeuralNetworkBuilder(2)
                .SetLossFunction(LossFunction.MeanSquaredError)
                .AddFCLayer(2, ActivationFunction.Sigmoid)
                .AddFCLayer(1, ActivationFunction.Sigmoid).Build();

            float[][] inputs = new float[4][];
            float[][] outputs = new float[4][];

            for(int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = new float[2];
                inputs[i][0] = (i & 1);
                inputs[i][1] = (i & 2) >> 1;

                outputs[i] = new float[1];
                outputs[i][0] = ((i & 1) ^ ((i & 2) >> 1));
            }

            GeneticTrainer trainer = new GeneticTrainer(nn, 0, 300);
            trainer.Train(inputs, outputs, inputs.Length, 100, 25, 10, 0.4f, 0.01f, 1500);

            for(int i = 0; i < inputs.Length; i++)
            {
                var o = nn.Forward(inputs[i]);

                Console.WriteLine($"Inputs = [{inputs[i][0]}, {inputs[i][1]}], Output = [{o[0]}], Expected = [{outputs[i][0]}], Loss = {nn.Loss(outputs[i])}");
            }*/
            var ann = new AnimeAutoencoder();
            ann.InitializeDataset();
            ann.Train();

            Console.ReadLine();
        }
    }
}
