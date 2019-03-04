using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.LossFunctions
{
    public class NamedLossFunction : LossFunctionBase
    {
        public const string GANDiscFake = "gan_disc_fake";
        public const string GANDiscReal = "gan_disc_real";
        public const string GANGen = "gan_gen";
        public const string BCE = "binary_cross_entropy";

        public NamedLossFunction(string lossName, string lossDeriv) : base(lossName, lossDeriv) { }
    }
}
