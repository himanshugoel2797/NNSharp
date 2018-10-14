namespace NNSharp2.ComputationalGraph
{
    public enum NodeOperationType
    {
        Gradient,

        //Complex operations - have their own shaders
        
        //Simple operations - can be added into single shaders
        MatrixDeclaration,
        VectorDeclaration,
        ConstantDeclaration,
        ConstantVectorDeclaration,
        ConstantMatrixDeclaration,

        Assignment,
        
        Transpose,
        MatrixProduct,
        TensorProduct,
        HadamardProduct,

        Add,
        Subtract,
        Multiply,
        Divide,
    }
}