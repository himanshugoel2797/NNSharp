using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IDataset
    {
        void Initialize();
        Vector LoadItem(string file);

        void GetNextTrainingSet(out Vector input, out Vector output);

        int GetInputSize();
        int GetOutputSize();
    }
}
