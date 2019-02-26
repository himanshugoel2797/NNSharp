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
        Matrix LoadItem(string file);

        void GetNextTrainingSet(out Matrix input, out Matrix output);

        int GetInputSize();
        int GetOutputSize();
    }
}
