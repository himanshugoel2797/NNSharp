using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Kernels;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.WeightInitializers;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeImageClassifier
{
    class Program
    {
        const int Side = 64;

        [STAThread]
        static void Main(string[] args)
        {
            KernelManager.Initialize();

            var tags = new string[]
            {
                //"emilia_(re:zero)",
                //"katou_megumi",
                //"tokisaki_kurumi",
                //"sawamura_spencer_eriri",
                //"kasumigaoka_utaha",
                //"zero_two_(darling_in_the_franxx)",
                //"nishikino_maki",
                //"souryuu_asuka_langley",
                //"shiba_miyuki",
                //"akemi_homura",
                //"kizuna_ai",
                //"euryale",
                "osakabe-hime_(fate/grand_order)",
                "momo_velia_deviluke",
                "semiramis_(fate)",
                "altera_(fate)",
                "takao_(aoki_hagane_no_arpeggio)",
                "medb_(fate)_(all)",
                "meltlilith",
                "yuzuriha_inori",
                //"redjuice"
            };

            var globalTags = new string[]
            {
                "1girl",
                "solo",
                "-*boy*",
                "-large_breasts",
                "-video",
            };

            BooruDatasetBuilder datasetBuilder = new BooruDatasetBuilder();
            for (int i = 0; i < globalTags.Length; i++)
                datasetBuilder.AddGlobalTag(globalTags[i]);

            for (int i = 0; i < tags.Length; i++)
                datasetBuilder.AddLocalTag(tags[i]);

            datasetBuilder.Download(500, @"I:\Datasets\Gelbooru");
            //var inputDataset = datasetBuilder.GetDataset(@"I:\Datasets\Gelbooru", @"I:\Datasets\Gelbooru_SMALL", Side, 250);
            /*
            var classifier = new NeuralNetworkBuilder(Side * Side * 3)
                                .WeightInitializer(new UniformWeightInitializer(0, 0))
                                .LossFunction<Quadratic>()
                                .AddConv(3, 3, 1, 0, Side, 3)
                                .AddActivation<ReLU>()
                                .AddPooling(2, 2, 3)
                                .AddConv(3, 10, 1, 0, Side / 2, 3)
                                .AddActivation<ReLU>()
                                .AddPooling(2, 2, 10)
                                .AddFC(4096)
                                .AddActivation<LeakyReLU>()
                                .AddFC(1024)
                                .AddActivation<ReLU>()
                                .AddFC(512)
                                .AddActivation<ReLU>()
                                .AddFC(256)
                                .AddActivation<ReLU>()
                                .AddFC(64)
                                .AddActivation<ReLU>()
                                .AddFC(16)
                                .AddActivation<ReLU>()
                                .AddFC(8)
                                .AddActivation<ReLU>()
                                .AddFC(8)
                                .AddActivation<ReLU>()
                                .AddFC(8)
                                .AddActivation<ReLU>()
                                .AddFC(tags.Length)
                                .AddActivation<Sigmoid>()
                                .Build();

            var trainer = new ClassifierTrainer("Anime Classifier", tags, classifier);
            trainer.SetDataset(inputDataset);

            LearningManager.Show(trainer);*/
        }
    }
}
