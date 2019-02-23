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
        void Optimize(Matrix w, Matrix nabla_w);
        void Optimize(Vector b, Vector nabla_b);
        void Update(float curError);
    }
}
