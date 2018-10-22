using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.ANN
{
    public class NeuralNetwork
    {
        public enum StateValue
        {
            Weights,
            Biases,
            Errors,
        }

        List<LayerDef> layers;
        struct LayerDef
        {
            public LayerType layerType;
            public Matrix[] cmn;
            public Matrix[] p;
            public Matrix[] p_shdw;
            public Shader fwd;
            public Shader bkwd_werr;
            public Shader bkwd;
            public Shader bkwd_wu;
        }

        public int InputSize { get { return layers[0].p[0].Columns; } }
        public int OutputSize { get { return layers.Last().cmn[0].Rows; } }
        public int LayerCount { get { return layers.Count; } }

        public NeuralNetwork()
        {
            layers = new List<LayerDef>();
        }

        internal void Add(LayerType layerType, Matrix[] cmn, Matrix[] p, Matrix[] p_shdw, Shader fwd, Shader bkwd_werr, Shader bkwd, Shader bkwd_wu)
        {
            layers.Add(new LayerDef()
            {
                layerType = layerType,
                cmn = cmn,
                p = p,
                p_shdw = p_shdw,
                fwd = fwd,
                bkwd_werr = bkwd_werr,
                bkwd = bkwd,
                bkwd_wu = bkwd_wu,
            });
        }

        internal void Add(NeuralNetwork nn)
        {
            for (int i = 0; i < nn.LayerCount; i++)
            {
                layers.Add(nn.layers[i]);
            }
        }

        #region Forward
        private static Matrix layerfwd(LayerDef d, Matrix i)
        {
            switch (d.layerType)
            {
                case LayerType.FC:
                    d.fwd.Set("w", d.p[0].tex, true, false);
                    d.fwd.Set("b", d.p[1].tex, true, false);
                    d.fwd.Set("i", i.tex, true, false);
                    d.fwd.Set("o", d.cmn[0].tex, false, true);
                    d.fwd.Set("a", d.cmn[1].tex, false, true);

                    d.fwd.Dispatch((uint)d.p[0].tex.Width, 1, 1);
                    return d.cmn[1];   
                      
                default:
                    throw new NotImplementedException(); 
            }
        }

        public Matrix Forward(Matrix i)
        {
            i = layerfwd(layers[0], i);
            for (int idx = 1; idx < layers.Count; idx++)
            {
                i = layerfwd(layers[idx], i);
            }
            return i;
        }
        #endregion

        #region Backward
        private static Matrix layerbkwd(LayerDef d, Matrix eo)
        {
            switch (d.layerType)
            {
                case LayerType.FC:
                    //Compute the werr if it's present
                    if(d.bkwd_werr != null)
                    {
                        d.bkwd_werr.Set("w", d.p[0].tex, true, false);
                        d.bkwd_werr.Set("eo", eo.tex, true, false);

                        d.bkwd_werr.Set("errO", d.cmn[2].tex, false, true);

                        d.bkwd_werr.Dispatch((uint)d.p[0].tex.Height, 1, 1);
                    }

                    d.bkwd.Set("w", d.p[0].tex, true, false);
                    d.bkwd.Set("b", d.p[1].tex, true, false);
                    d.bkwd.Set("o", d.cmn[0].tex, true, false);
                    d.bkwd.Set("a", d.cmn[1].tex, true, false);
                    d.bkwd.Set("eo", eo.tex, true, false);
                    
                    d.bkwd.Set("errO", d.cmn[2].tex, true, true);

                    d.bkwd.Dispatch((uint)d.p[0].tex.Width, 1, 1);
                    return d.cmn[2];
                default:
                    throw new NotImplementedException();
            }
        }

        private static Matrix layerbkwd_weights_update(LayerDef d, Matrix i, float learning_rate)
        {
            switch (d.layerType)
            {
                case LayerType.FC:
                    d.bkwd_wu.Set("i", i.tex, true, false);
                    d.bkwd_wu.Set("err", d.cmn[2].tex, true, false);
                    d.bkwd_wu.Set("w", d.p[0].tex, true, true);
                    d.bkwd_wu.Set("b", d.p[1].tex, true, true);

                    d.bkwd_wu.Set("learning_rate", learning_rate);

                    d.bkwd_wu.Dispatch((uint)d.p[0].tex.Width, (uint)d.p[0].tex.Height, 1);
                    return d.cmn[1];
                default:
                    throw new NotImplementedException();
            }
        }

        public Matrix Backward(Matrix eo)
        {
            Matrix eo_l = eo;
            for (int idx = LayerCount - 1; idx >= 0; idx--)
            {
                eo_l = layerbkwd(layers[idx], eo_l);
            }

            return eo_l;
        }

        public void UpdateWeights(Matrix i, float learning_rate)
        {
            //Now we have error vectors for each layer, update the weights and biases
            Matrix in_l = i;
            for (int idx = 0; idx < LayerCount; idx++)
            {
                in_l = layerbkwd_weights_update(layers[idx], in_l, learning_rate);
            }
        }
        #endregion

        public (float[], int, int) State(int i, StateValue stateStringValue)
        {
            //Read in the weights, biases and errors
            int r_idx = -1;
            switch (stateStringValue)
            {
                case StateValue.Weights:
                    r_idx = 0;
                    break;
                case StateValue.Biases:
                    r_idx = 1;
                    break;
                case StateValue.Errors:
                    r_idx = 2;
                    break;

            }

            if (r_idx < 2)
            {
                float[] data = new float[layers[i].p[r_idx].tex.Height * layers[i].p[r_idx].tex.Width];
                layers[i].p[r_idx].Read(data);
                return (data, layers[i].p[r_idx].Rows, layers[i].p[r_idx].Columns);
            }
            else
            {
                float[] data = new float[layers[i].cmn[r_idx].tex.Height * layers[i].cmn[r_idx].tex.Width];
                layers[i].cmn[r_idx].Read(data);
                return (data, layers[i].cmn[r_idx].Rows, layers[i].cmn[r_idx].Columns);
            }
        }
    }
}
