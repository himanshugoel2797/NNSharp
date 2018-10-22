using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IOptimizer
    {
        void SetLearningRate(float v);
        void Optimize(Matrix w, Vector b, Matrix nabla_w, Vector nabla_b);
    }
}
