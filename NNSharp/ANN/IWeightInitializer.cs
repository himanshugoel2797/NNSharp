using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IWeightInitializer
    {
        float GetBias();
        float GetWeight(int in_dim, int out_dim);
    }
}
