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
        List<LayerDef> layers;
        struct LayerDef
        {
            public LayerType layerType;
            public Matrix[] cmn;
            public Matrix[] p;
            public Matrix[] p_shdw;
            public Shader fwd;
            public Shader bkwd;
        }

        public NeuralNetwork()
        {
            layers = new List<LayerDef>();
        }

        public void Add(LayerType layerType, Matrix[] cmn, Matrix[] p, Matrix[] p_shdw, Shader fwd, Shader bkwd)
        {
            layers.Add(new LayerDef()
            {
                layerType = layerType,
                cmn = cmn,
                p = p,
                p_shdw = p_shdw,
                fwd = fwd,
                bkwd = bkwd,
            });
        }

        #region Forward
        private static Matrix layerfwd(LayerDef d, Matrix i)
        {
            switch (d.layerType)
            {
                case LayerType.FC:
                    d.fwd.Set("i", i.tex, true, false);
                    d.fwd.Set("w", d.p[0].tex, true, false);
                    d.fwd.Set("b", d.p[1].tex, true, false);
                    d.fwd.Set("o", d.cmn[0].tex, false, true);
                    d.fwd.Set("a", d.cmn[1].tex, false, true);

                    uint cols = (uint)d.cmn[1].Columns;
                    if (cols % 4 != 0)
                        cols += 4 - (cols % 4);
                     
                    d.fwd.Dispatch(cols / 4, 1, 1);
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
        private static Matrix layerbkwd(LayerDef d, Matrix i)
        {
            switch (d.layerType)
            {
                case LayerType.FC:
                    d.fwd.Set("i", i.tex, true, false);
                    d.fwd.Set("w", d.p[0].tex, true, false);
                    d.fwd.Set("b", d.p[1].tex, true, false);
                    d.fwd.Set("o", d.cmn[0].tex, false, true);
                    d.fwd.Set("a", d.cmn[1].tex, false, true);
                    return d.cmn[1];

                default:
                    throw new NotImplementedException();
            }
        }

        public Matrix Backward(Matrix i)
        {
            i = layerfwd(layers[0], i);
            for (int idx = 1; idx < layers.Count; idx++)
            {
                i = layerfwd(layers[idx], i);
            }
            return i;
        }
        #endregion
    }
}
