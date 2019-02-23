using NNSharp.ANN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NNSharp.Tools
{
    public class LearningManager
    {
        static LearningProgressForm form;

        public static void Show(INetworkTrainer trainer)
        {
            var n = new LearningProgressForm();
            n.LoadNetwork(trainer);
            n.ShowDialog();
        }
    }
}
