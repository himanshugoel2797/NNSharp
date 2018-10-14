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

        public void Optimize(Matrix w, Vector b, Matrix nabla_w, Vector nabla_b, Matrix w_dst, Vector b_dst)
        {
            Vector.MSub(nabla_b, b, r, b_dst);
            Matrix.MSub(nabla_w, w, r, w_dst);
        }

        public void SetLearningRate(float v)
        {
            r = v;
        }
    }
}
