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
        private float r;

        public SGD()
        {
        }

        public void Optimize(ILayer layer, int idx, Matrix w, Matrix nabla_w)
        {
            Matrix.MSubSelf(nabla_w, w, r);
        }

        public void Optimize(ILayer layer, int idx, Vector b, Vector nabla_b)
        {
            Vector.MSubSelf(nabla_b, b, r);
        }

        public void RegisterLayer(ILayer layer, int w_cnt, int ww_len, int wh_len, int b_cnt, int b_len)
        {

        }

        public void SetLearningRate(float v)
        {
            r = v;
        }

        public void Update(float curError)
        {

        }
    }
}
