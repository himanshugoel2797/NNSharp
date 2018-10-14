using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface ILossFunction
    {
        void Loss(Vector output, Vector expectedOutput, Vector result);
        void LossDeriv(Vector output, Vector expectedOutput, Vector result);
    }
}
