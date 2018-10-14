using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    [Serializable]
    public class GAN_GenCrossEntropy : LossFunctionBase
    {
        public GAN_GenCrossEntropy() : base("gan_gen_crossentropy_loss", "gan_gen_crossentropy_loss_deriv") { }
    }
}
