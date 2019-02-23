using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface INetworkTrainer
    {
        //Autoencoder trainer
        //Label trainer
        //Adversarial trainer

        float LearningRate { get; set; }

        int OutputSeriesCount();

        bool RunIteration(int iter, out double[] loss);
        void Test(string filename);

        void SetDataset(IDataset dataset);
        void Save(string filename);
    }
}
