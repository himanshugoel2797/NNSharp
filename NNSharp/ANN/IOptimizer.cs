using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IOptimizer
    {
        void RegisterLayer(ILayer layer, int w_cnt, int ww_len, int wh_len, int b_cnt, int b_len);
        void OptimizeWeights(ILayer layer, int idx, Matrix w, Matrix nabla_w);
        void OptimizeBiases(ILayer layer, int idx, Matrix w, Matrix nabla_w);
        void Update(float curError);
    }
}
