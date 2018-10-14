using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IWeightInitializable
    {
        void SetWeights(IWeightInitializer weightInitializer);
    }
}
