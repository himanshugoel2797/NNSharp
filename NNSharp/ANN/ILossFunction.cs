using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface ILossFunction
    {
        void Loss(Matrix output, Matrix expectedOutput, Matrix result, float regularizationParam);
        void LossDeriv(Matrix output, Matrix expectedOutput, Matrix result, float regularizationParam);
    }
}
