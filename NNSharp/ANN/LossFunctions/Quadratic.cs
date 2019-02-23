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
#if CPU
        public Quadratic() : base("quadratic_loss", "quadratic_loss_deriv") { }
#elif GPU
        public Quadratic() : base("activ_res = 0.5f * (o - eo) * (o - eo);", "activ_res = (o - eo);") { }
#endif
    }
}
