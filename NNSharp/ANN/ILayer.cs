using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface ILayer
    {
        Vector[] Forward(params Vector[] input);

        Vector[] Propagate(params Vector[] prev_delta);
        Vector[] GetLastDelta();

        void LayerError(params Vector[] prev_delta);
        void Learn(IOptimizer opt);
        void ResetLayerError();

        #region Parameter Setup
        int GetOutputSize();
        int GetOutputDepth();
        void SetInputSize(int input_side, int input_depth);
        #endregion
    }
}
