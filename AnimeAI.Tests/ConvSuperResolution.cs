using NNSharp;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.ANN.NetworkBuilder;
using NNSharp.Tools;
using System;
using System.IO;
using System.Linq;

namespace AnimeAI.Tests
{
    class ConvSuperResolution
    {
        LayerContainer superres_enc_front, superres_enc_back, superres_dec_front, superres_dec_back;

        const int StartSide = 96;
        const int LatentSize = 6 * 6 * 32;
        const int EndSide = 96;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 16;

        public ConvSuperResolution()
        {
            superres_enc_front = InputLayer.Create(StartSide, 3);
            superres_enc_back = ActivationLayer.Create<ReLU>();

            var pooling_0 = PoolingLayer.Create(2, 2);
            var pooling_1 = PoolingLayer.Create(2, 2);
            var pooling_2 = PoolingLayer.Create(2, 2);
            var pooling_3 = PoolingLayer.Create(2, 2);

            superres_enc_front.Append(
                ConvLayer.Create(5, 128, 2).Append(             //o = 96
                ActivationLayer.Create<ReLU>().Append(
                    pooling_0.Append(                           //o = 48
                ConvLayer.Create(3, 128, 1).Append(              //o = 48
                ActivationLayer.Create<ReLU>().Append(
                    pooling_1.Append(                           //o = 24
                ConvLayer.Create(3, 64, 1).Append(              //o = 24
                ActivationLayer.Create<ReLU>().Append(
                    pooling_2.Append(                           //o = 12
                ConvLayer.Create(3, 32, 1).Append(              //o = 12
                ActivationLayer.Create<ReLU>().Append(
                    pooling_3.Append(                           //o = 6
                ConvLayer.Create(3, 32, 1).Append(              //o = 6
                superres_enc_back
            ))))))))))))));

            superres_dec_front = InputLayer.Create(6, 32);
            superres_dec_back = ActivationLayer.Create<Tanh>();

            superres_dec_front.Append(
                ConvLayer.Create(3, 32, 1).Append(              //o = 6
                ActivationLayer.Create<ReLU>().Append(
                    UnpoolingLayer.Create(pooling_3).Append(    //o = 12
                ConvLayer.Create(3, 64, 1).Append(              //o = 12
                ActivationLayer.Create<ReLU>().Append(
                    UnpoolingLayer.Create(pooling_2).Append(    //o = 24
                ConvLayer.Create(3, 128, 1).Append(              //o = 24
                ActivationLayer.Create<ReLU>().Append(
                    UnpoolingLayer.Create(pooling_1).Append(    //o = 48
                ConvLayer.Create(3, 128, 1).Append(             //o = 48
                ActivationLayer.Create<ReLU>().Append(
                    UnpoolingLayer.Create(pooling_0).Append(    //o = 96
                ConvLayer.Create(5, 3, 2).Append(               //o = 96
                    superres_dec_back
            ))))))))))))));

            superres_enc_back.Append(superres_dec_front);

            //TODO: come up with an approach that saves the convolution/multiplication indexes and rearranges the weights etc so they fit into cache better
            //TODO: unpooling layer tied to pooling layers

            //Initialize Weights
            superres_enc_front.SetupInternalState();
            superres_enc_front.InitializeWeights(new UniformWeightInitializer(0, 0.001f));
        }

        public void Train()
        {
            string dir = "ND_OPT_ConvAutoencoder_Data";

            Directory.CreateDirectory($@"{dir}");
            Directory.CreateDirectory($@"{dir}\Results");
            Directory.CreateDirectory($@"{dir}\Sources");

            AnimeDatasets a_dataset = new AnimeDatasets(StartSide, @"I:\Datasets\VAE_Dataset\White", @"I:\Datasets\VAE_Dataset\White\conv");//@"I:\Datasets\anime-faces\combined", @"I:\Datasets\anime-faces\combined_small");
            a_dataset.InitializeDataset();

            AnimeDatasets b_dataset = new AnimeDatasets(EndSide, @"I:\Datasets\VAE_Dataset\White", @"I:\Datasets\VAE_Dataset\White\conv");//@"I:\Datasets\anime-faces\combined", @"I:\Datasets\anime-faces\combined_small");
            b_dataset.InitializeDataset();

            Adam sgd = new Adam(0.001f);
            Quadratic quadratic = new Quadratic();

            NRandom r = new NRandom(0);
            NRandom r2 = new NRandom(0);

            Matrix loss_deriv = new Matrix(OutputSize, 1, MemoryFlags.ReadWrite, true);


            #region Setup Database
            Matrix data_vec = new Matrix(LatentSize, 1, MemoryFlags.ReadOnly, false);

            Matrix[] a_dataset_vec = new Matrix[a_dataset.TrainingFiles.Count];
            float[][] a_dataset_f = new float[a_dataset.TrainingFiles.Count][];

            Matrix[] b_dataset_vec = new Matrix[a_dataset.TrainingFiles.Count];
            float[][] b_dataset_f = new float[a_dataset.TrainingFiles.Count][];

            for (int i = 0; i < a_dataset.TrainingFiles.Count; i++)
            {
                a_dataset_f[i] = new float[InputSize];
                a_dataset_vec[i] = new Matrix(InputSize, 1, MemoryFlags.ReadOnly, false);
                a_dataset.LoadImage(a_dataset.TrainingFiles[i], a_dataset_f[i]);
                a_dataset_vec[i].Write(a_dataset_f[i]);

                b_dataset_f[i] = new float[OutputSize];
                b_dataset_vec[i] = new Matrix(OutputSize, 1, MemoryFlags.ReadOnly, false);
                b_dataset.LoadImage(b_dataset.TrainingFiles[i], b_dataset_f[i]);
                b_dataset_vec[i].Write(b_dataset_f[i]);
            }
            #endregion

            for (int i0 = 000; i0 < 20000 * BatchSize; i0++)
            {
                int idx = (r.Next() % (a_dataset.TrainingFiles.Count / 2));

                var out_img = superres_enc_front.ForwardPropagate(a_dataset_vec[idx]);
                quadratic.LossDeriv(out_img[0], b_dataset_vec[idx], loss_deriv, 0);

                superres_dec_back.ResetLayerErrors();
                superres_dec_back.ComputeGradients(loss_deriv);
                superres_dec_back.ComputeLayerErrors(loss_deriv);
                superres_dec_back.UpdateLayers(sgd);

                loss_deriv.Clear();

                if (i0 % BatchSize == 0)
                {
                    a_dataset.SaveImage($@"{dir}\Sources\{i0 / BatchSize}.png", a_dataset_f[idx]);
                    b_dataset.SaveImage($@"{dir}\Results\{i0 / BatchSize}.png", out_img[0].Read());
                }

                Console.Clear();
                Console.Write($"Iteration: {i0 / BatchSize}, Sub-Batch: {i0 % BatchSize}");
            }

            superres_enc_front.Save($@"{dir}\network_final.bin");
            Console.WriteLine("DONE.");
        }
    }
}
