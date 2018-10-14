using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    [Serializable]
    public class Quadratic : LossFunctionBase
    {
        public Quadratic() : base("quadratic_loss", "quadratic_loss_deriv") { }
    }
}
