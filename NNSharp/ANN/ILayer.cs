using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface ILayer
    {
        Matrix[] Forward(params Matrix[] input);

        Matrix[] Propagate(params Matrix[] prev_delta);
        Matrix[] GetLastDelta();

        void LayerError(params Matrix[] prev_delta);
        void Learn(IOptimizer opt);
        void ResetLayerError();

        #region Parameter Setup
        int GetOutputSize();
        int GetOutputDepth();
        void SetInputSize(int input_side, int input_depth);
        #endregion
    }
}
