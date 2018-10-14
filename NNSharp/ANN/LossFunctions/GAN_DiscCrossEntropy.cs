using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    [Serializable]
    public class GAN_DiscCrossEntropy : LossFunctionBase
    {
        public GAN_DiscCrossEntropy() : base("gan_disc_crossentropy_loss", "gan_disc_crossentropy_loss_deriv") { }
    }
}
