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
        void SetOptimizer(IOptimizer opt);
        IOptimizer GetOptimizer();

        int GetOutputSize(int input);
        void Learn();
        void Error(Vector input, Vector prev_delta, Matrix prev_w, out Vector cur_delta, out Matrix cur_w);

        Vector Forward(Vector input);
    }
}
