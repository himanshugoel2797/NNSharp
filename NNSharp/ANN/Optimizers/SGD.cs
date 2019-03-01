using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Optimizers
{
    [Serializable]
    public class SGD : IOptimizer
    {
        private readonly float r;

        public SGD(float learning_rate)
        {
            r = learning_rate;
        }

        public void OptimizeWeights(ILayer layer, int idx, Matrix w, Matrix nabla_w)
        {
            Matrix.Fmop(nabla_w, -r, w, 1, w);
        }

        public void OptimizeBiases(ILayer layer, int idx, Matrix b, Matrix nabla_b)
        {
            Matrix.Fmop(nabla_b, -r, b, 1, b);
        }

        public void RegisterLayer(ILayer layer, int w_cnt, int ww_len, int wh_len, int b_cnt, int b_len)
        {

        }

        public void Update(float curError)
        {

        }
    }
}
