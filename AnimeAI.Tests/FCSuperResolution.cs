﻿using NNSharp;
using NNSharp.ANN;
using NNSharp.ANN.ActivationFunctions;
using NNSharp.ANN.Layers;
using NNSharp.ANN.LossFunctions;
using NNSharp.ANN.Optimizers;
using NNSharp.ANN.WeightInitializers;
using NNSharp.ANN.NetworkBuilder;
using NNSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    class FCSuperResolution
    {
        LayerContainer superres_enc_front, superres_enc_back, superres_dec_front, superres_dec_back;

        const int StartSide = 32;
        const int LatentSize = 4 * 4 * 4;
        const int EndSide = 32;
        const int InputSize = StartSide * StartSide * 3;
        const int OutputSize = EndSide * EndSide * 3;

        const int BatchSize = 5;

        public FCSuperResolution()
        {
            superres_enc_front = InputLayer.Create(StartSide, 3);
            superres_enc_back = ActivationLayer.Create<LeakyReLU>();

            superres_enc_front.Append(
                FCLayer.Create(16, 16).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(8, 8).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(4, 4).Append(
                    superres_enc_back
            ))))));

            superres_dec_front = InputLayer.Create(4, 4);
            superres_dec_back = ActivationLayer.Create<Tanh>();

            superres_dec_front.Append(
                FCLayer.Create(8, 8).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(16, 8).Append(
                ActivationLayer.Create<LeakyReLU>().Append(
                FCLayer.Create(32, 3).Append(
                    superres_dec_back
            ))))));

            superres_enc_back.Append(superres_dec_front);

            //Initialize Weights
            superres_enc_front.SetupInternalState();
            superres_enc_front.InitializeWeights(new UniformWeightInitializer(0, 0.001f));
        }

        public void Train()
        {
            string dir = "NDFCAutoencoder_Data";

            Directory.CreateDirectory($@"{dir}");
            Directory.CreateDirectory($@"{dir}\Results");
            Directory.CreateDirectory($@"{dir}\Sources");

            AnimeDatasets a_dataset = new AnimeDatasets(StartSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_small");
            a_dataset.InitializeDataset();

            AnimeDatasets b_dataset = new AnimeDatasets(EndSide, @"I:\Datasets\anime-faces\emilia_(re_zero)", @"I:\Datasets\anime-faces\emilia_small");
            b_dataset.InitializeDataset();

            Adam sgd = new Adam(0.0001f);
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

            for (int i0 = 000; i0 < 10000 * BatchSize; i0++)
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
                Console.Write($"Iteration: {i0}");
            }

            superres_enc_front.Save($@"{dir}\network_final.bin");
            Console.WriteLine("DONE.");
        }
    }
}
