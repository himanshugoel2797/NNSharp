using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.NetworkTrainer
{
    public class AutoencoderTrainer : INetworkTrainer
    {
        public float LearningRate { get { return learningRate; } set { learningRate = value; optimizer.SetLearningRate(value); } }

        private string trainerName;
        private IDataset dataset;
        private float learningRate = 0.1f;
        private float errorAvg = 0;

        private NeuralNetwork autoencoder, encoder, decoder;
        private SGD optimizer;

        public AutoencoderTrainer(string name, NeuralNetwork encoder, NeuralNetwork decoder)
        {
            trainerName = name;

            autoencoder = new NeuralNetworkBuilder(encoder.InputSize)
                                .LossFunction<Quadratic>()
                                .Add(encoder)
                                .Add(decoder)
                                .Build();

            this.encoder = encoder;
            this.decoder = decoder;

            optimizer = new SGD();
            optimizer.SetLearningRate(learningRate);
        }

        public bool RunIteration(int iter, out double[] loss)
        {
            dataset.GetNextTrainingSet(out var input, out var output);
            autoencoder.TrainSingle(input, output, optimizer);

            errorAvg += autoencoder.Error();

            loss = new double[1];
            if (iter % 100 == 0)
            {
                loss[0] = errorAvg / 100;
                errorAvg = 0;

                if (double.IsNaN(loss[0]) | double.IsInfinity(loss[0]))
                    return false;
                return true;
            }
            return false;
        }

        public void Save(string filename)
        {
            encoder.Save(filename + ".enc");
            decoder.Save(filename + ".dec");
        }

        public void SetDataset(IDataset dataset)
        {
            this.dataset = dataset;
        }

        public void Test(string filename)
        {
            var input = (dataset as Datasets.NoisyImageSet).LoadItem(filename);
            var output = autoencoder.Forward(input);
            ImageManipulation.SaveImage("test_input.png", input, (int)Math.Sqrt(input.Length / 3));
            ImageManipulation.SaveImage("test_output.png", output, (int)Math.Sqrt(output.Length / 3));
        }

        public override string ToString()
        {
            return trainerName;
        }

        public int OutputSeriesCount()
        {
            return 1;
        }
    }
}
