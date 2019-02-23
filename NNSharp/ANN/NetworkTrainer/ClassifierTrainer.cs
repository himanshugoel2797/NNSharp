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
    public class ClassifierTrainer : INetworkTrainer
    {
        public float LearningRate { get { return learningRate; } set { learningRate = value; optimizer.SetLearningRate(value); } }

        private string trainerName;
        private IDataset dataset;
        private float learningRate = 0.1f;
        private float errorAvg = 0;

        private NeuralNetwork classifier;
        private SGD optimizer;
        private string[] labels;

        public ClassifierTrainer(string name, string[] lbls, NeuralNetwork classifier)
        {
            trainerName = name;
            this.classifier = classifier;
            labels = lbls;

            optimizer = new SGD();
            optimizer.SetLearningRate(learningRate);
        }

        public bool RunIteration(int iter, out double[] loss)
        {
            dataset.GetNextTrainingSet(out var input, out var output);
            classifier.TrainSingle(input, output, optimizer);

            errorAvg += classifier.Error();

            loss = new double[1];
            if (iter % 50 == 0)
            {
                loss[0] = errorAvg / 50;
                errorAvg = 0;

                if (double.IsNaN(loss[0]) | double.IsInfinity(loss[0]))
                    return false;
                return true;
            }
            return false;
        }

        public void Save(string filename)
        {
            classifier.Save(filename + ".bin");
        }

        public void SetDataset(IDataset dataset)
        {
            this.dataset = dataset;
        }

        public void Test(string filename)
        {
            var input = dataset.LoadItem(filename);
            var output = classifier.Forward(input);
            ImageManipulation.SaveImage("test_input.png", input, (int)Math.Sqrt(input.Length / 3));

            float[] lbls = new float[labels.Length];
            output.Read(lbls);

            using (var f = File.OpenWrite("test_output.txt"))
            using (var f_sw = new StreamWriter(f))
            {
                for (int i = 0; i < labels.Length; i++)
                    f_sw.WriteLine($"{labels[i]} : {lbls[i]}");
            }
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
