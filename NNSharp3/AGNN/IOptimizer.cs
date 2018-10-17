using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.AGNN
{
    public interface IOptimizer
    {
        void SetLearningRate(float v);
        //void Optimize(Matrix w, Vector b, Matrix nabla_w, Vector nabla_b, Matrix w_dst, Vector b_dst);
    }
}
