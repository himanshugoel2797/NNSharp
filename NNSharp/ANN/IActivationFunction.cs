﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public interface IActivationFunction
    {
        ActivationFunctionInfo Activation();
        ActivationFunctionInfo DerivActivation();
    }
}
