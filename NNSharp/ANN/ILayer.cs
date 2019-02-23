using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface ILayer
    {
        void SetInputSize(int sz);

        int GetOutputSize(int input);
        void Learn(IOptimizer opt);
        void Reset();

        Vector Error(Vector prev_delta, bool update_cur);

        Vector Forward(Vector input);
    }
}
