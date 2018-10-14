using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SMath = System.Math;

namespace NNSharp2.ComputationalGraph.Compiler
{
    class GraphCompiler
    {
        static List<Node> roots;

        static GraphCompiler()
        {
            roots = new List<Node>();
        }

        public static void AddRoot(Node n)
        {
            roots.Add(n);
        }

        static int rNodeIdx = 0;
        public static GraphContext Build()
        {
            nodes = new List<Node>();
            subtrees = new List<Node>();

            GraphContext graphContext = new GraphContext();

            List<SingleGraph> graphs = new List<SingleGraph>();
            for (int i = 0; i < roots.Count; i++)
            {
                rNodeIdx = i;
                graphs.Add(BuildSingle(roots[i]));
            }
            roots.Clear();

            //Find all the common subtrees and update aliases
            var orderedCalls = SingleGraph.GetOrderedSubtreeGraphs();
            for (int i = 0; i < orderedCalls.Length; i++)
            {
                Console.WriteLine(orderedCalls[i].Name + "[" + orderedCalls[i].Subtree.Dimension[0] + ", " + orderedCalls[i].Subtree.Dimension[1] + "] = " + orderedCalls[i].Subtree);
                Console.WriteLine();
            }

            //TODO: matrix multiplications are then converted into subtrees, so they can evaluated separately

            //graph context contains the ordered calls, the input variables, the subtrees as intermediate variables, and the roots as output variables


            return graphContext;
        }

        #region Tree Optimization
        private static void SwapExpr(Node root)
        {
            //Swap nodes such that the constant nodes are on the left
            for (int i = 0; i < root.IncomingEdges.Count; i++)
                SwapExpr(root.IncomingEdges[i]);

            if (root.IncomingEdges.Count == 2)
            {
                if ((root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[0].Operation != NodeOperationType.ConstantDeclaration) |
                    (root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[0].Operation != NodeOperationType.ConstantMatrixDeclaration))
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Add:
                        case NodeOperationType.Multiply:
                        case NodeOperationType.Hadamard:
                            {
                                var tmp = root.IncomingEdges[0];
                                root.IncomingEdges[0] = root.IncomingEdges[1];
                                root.IncomingEdges[1] = tmp;
                            }
                            break;
                    }
                }
            }
        }

        private static void ConstExpr(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
                ConstExpr(root.IncomingEdges[i]);

            if (root.IncomingEdges.Count == 1)
            {
                if (root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Log:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Log(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Tanh:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Tanh(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Exp:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Exp(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                    }
                }
            }
            else if (root.IncomingEdges.Count == 2)
            {
                if (root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Add:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] + root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Subtract:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] - root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Multiply:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] * root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Divide:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] / root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Power:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Pow(root.IncomingEdges[0].OpSpecificValue[0], root.IncomingEdges[1].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                    }
                }
                else if ((root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 0) || (root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 0))
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Multiply:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { 0 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 0 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 0)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Divide:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { 0 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Add:
                            {
                                root.Operation = root.IncomingEdges[1].Operation;
                                root.OpSpecificValue = root.IncomingEdges[1].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[1].VariableName;

                                var r_in = root.IncomingEdges[1];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                        case NodeOperationType.Subtract:
                            {
                                root.IncomingEdges[0].OpSpecificValue[0] = -1;
                                root.Operation = NodeOperationType.Multiply;
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 0)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Power:
                            {
                                root.Operation = NodeOperationType.ConstantDeclaration;
                                root.OpSpecificValue = new double[] { 1 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Add:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 0 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Multiply:
                            {
                                root.Operation = root.IncomingEdges[1].Operation;
                                root.OpSpecificValue = root.IncomingEdges[1].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[1].VariableName;

                                var r_in = root.IncomingEdges[1];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Divide:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                        case NodeOperationType.Multiply:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Power:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }
            }
        }

        private static void ConstExpr2(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
                ConstExpr2(root.IncomingEdges[i]);

            if (root.IncomingEdges.Count == 2 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration)
            {
                if (root.IncomingEdges[1].IncomingEdges.Count == 2 && root.IncomingEdges[1].IncomingEdges[0].Operation == NodeOperationType.ConstantDeclaration)
                    switch (root.Operation)
                    {
                        case NodeOperationType.Add:
                            {
                                if (root.IncomingEdges[1].Operation == NodeOperationType.Add)
                                {
                                    root.IncomingEdges[0].Operation = NodeOperationType.ConstantDeclaration;
                                    root.IncomingEdges[0].OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] + root.IncomingEdges[1].IncomingEdges[0].OpSpecificValue[0] };
                                    root.IncomingEdges[0].IncomingEdges.Clear();

                                    root.IncomingEdges[1].VariableName = root.IncomingEdges[1].IncomingEdges[1].VariableName;
                                    root.IncomingEdges[1].Operation = root.IncomingEdges[1].IncomingEdges[1].Operation;
                                    root.IncomingEdges[1].OpSpecificValue = root.IncomingEdges[1].IncomingEdges[1].OpSpecificValue;
                                    root.IncomingEdges[1].IncomingEdges.Clear();
                                    root.IncomingEdges[1].IncomingEdges.Add(root.IncomingEdges[1].IncomingEdges[1]);
                                }
                            }
                            break;
                        case NodeOperationType.Multiply:
                            {
                                if (root.IncomingEdges[1].Operation == NodeOperationType.Multiply)
                                {
                                    root.IncomingEdges[0].Operation = NodeOperationType.ConstantDeclaration;
                                    root.IncomingEdges[0].OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] * root.IncomingEdges[1].IncomingEdges[0].OpSpecificValue[0] };
                                    root.IncomingEdges[0].IncomingEdges.Clear();

                                    root.IncomingEdges[1].VariableName = root.IncomingEdges[1].IncomingEdges[1].VariableName;
                                    root.IncomingEdges[1].Operation = root.IncomingEdges[1].IncomingEdges[1].Operation;
                                    root.IncomingEdges[1].OpSpecificValue = root.IncomingEdges[1].IncomingEdges[1].OpSpecificValue;
                                    var inc = root.IncomingEdges[1].IncomingEdges[1];
                                    root.IncomingEdges[1].IncomingEdges.Clear();
                                    root.IncomingEdges[1] = inc;
                                }
                            }
                            break;
                    }
            }
        }

        private static void ConstMatExpr(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
                ConstMatExpr(root.IncomingEdges[i]);

            if (root.IncomingEdges.Count == 1)
            {
                if (root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Log:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Log(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Tanh:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Tanh(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Exp:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Exp(root.IncomingEdges[0].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                    }
                }
            }
            else if (root.IncomingEdges.Count == 2)
            {
                if (root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Add:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] + root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Subtract:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] - root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Hadamard:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] * root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Divide:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] / root.IncomingEdges[1].OpSpecificValue[0] };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Power:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { SMath.Pow(root.IncomingEdges[0].OpSpecificValue[0], root.IncomingEdges[1].OpSpecificValue[0]) };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Multiply:
                            {
                                throw new Exception();
                            }
                            break;
                    }
                }
                else if ((root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 0) || (root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 0))
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Hadamard:
                        case NodeOperationType.Multiply:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { 0 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 0 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 0)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Divide:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { 0 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Add:
                            {
                                root.Operation = root.IncomingEdges[1].Operation;
                                root.OpSpecificValue = root.IncomingEdges[1].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[1].VariableName;

                                var r_in = root.IncomingEdges[1];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                        case NodeOperationType.Subtract:
                            {
                                root.IncomingEdges[0].OpSpecificValue[0] = -1;
                                root.Operation = NodeOperationType.Hadamard;
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 0)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Power:
                            {
                                root.Operation = NodeOperationType.ConstantMatrixDeclaration;
                                root.OpSpecificValue = new double[] { 1 };
                                root.IncomingEdges.Clear();
                            }
                            break;
                        case NodeOperationType.Add:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 0 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[0].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Hadamard:
                        //case NodeOperationType.Multiply:
                            {
                                root.Operation = root.IncomingEdges[1].Operation;
                                root.OpSpecificValue = root.IncomingEdges[1].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[1].VariableName;

                                var r_in = root.IncomingEdges[1];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Divide:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                        case NodeOperationType.Hadamard:
                        //case NodeOperationType.Multiply:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }

                if (root.IncomingEdges.Count > 1 && root.IncomingEdges[1].Operation == NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[1].OpSpecificValue[0] == 1)
                {
                    switch (root.Operation)
                    {
                        case NodeOperationType.Power:
                            {
                                root.Operation = root.IncomingEdges[0].Operation;
                                root.OpSpecificValue = root.IncomingEdges[0].OpSpecificValue;
                                root.VariableName = root.IncomingEdges[0].VariableName;

                                var r_in = root.IncomingEdges[0];
                                root.IncomingEdges.Clear();
                                root.IncomingEdges.AddRange(r_in.IncomingEdges);
                            }
                            break;
                    }
                }
            }
        }

        private static void ConstMatExpr2(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
                ConstMatExpr2(root.IncomingEdges[i]);

            if (root.IncomingEdges.Count == 2 && root.IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration)
            {
                if (root.IncomingEdges[1].IncomingEdges.Count == 2 && root.IncomingEdges[1].IncomingEdges[0].Operation == NodeOperationType.ConstantMatrixDeclaration)
                    switch (root.Operation)
                    {
                        case NodeOperationType.Add:
                            {
                                if (root.IncomingEdges[1].Operation == NodeOperationType.Add)
                                {
                                    root.IncomingEdges[0].Operation = NodeOperationType.ConstantMatrixDeclaration;
                                    root.IncomingEdges[0].OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] + root.IncomingEdges[1].IncomingEdges[0].OpSpecificValue[0] };
                                    root.IncomingEdges[0].IncomingEdges.Clear();

                                    root.IncomingEdges[1].VariableName = root.IncomingEdges[1].IncomingEdges[1].VariableName;
                                    root.IncomingEdges[1].Operation = root.IncomingEdges[1].IncomingEdges[1].Operation;
                                    root.IncomingEdges[1].OpSpecificValue = root.IncomingEdges[1].IncomingEdges[1].OpSpecificValue;
                                    root.IncomingEdges[1].IncomingEdges.Clear();
                                    root.IncomingEdges[1].IncomingEdges.Add(root.IncomingEdges[1].IncomingEdges[1]);
                                }
                            }
                            break;
                        case NodeOperationType.Hadamard:
                            {
                                if (root.IncomingEdges[1].Operation == NodeOperationType.Multiply)
                                {
                                    root.IncomingEdges[0].Operation = NodeOperationType.ConstantMatrixDeclaration;
                                    root.IncomingEdges[0].OpSpecificValue = new double[] { root.IncomingEdges[0].OpSpecificValue[0] * root.IncomingEdges[1].IncomingEdges[0].OpSpecificValue[0] };
                                    root.IncomingEdges[0].IncomingEdges.Clear();

                                    root.IncomingEdges[1].VariableName = root.IncomingEdges[1].IncomingEdges[1].VariableName;
                                    root.IncomingEdges[1].Operation = root.IncomingEdges[1].IncomingEdges[1].Operation;
                                    root.IncomingEdges[1].OpSpecificValue = root.IncomingEdges[1].IncomingEdges[1].OpSpecificValue;
                                    var inc = root.IncomingEdges[1].IncomingEdges[1];
                                    root.IncomingEdges[1].IncomingEdges.Clear();
                                    root.IncomingEdges[1] = inc;
                                }
                            }
                            break;
                        case NodeOperationType.Multiply:
                            {
                                throw new Exception();
                            }
                            break;
                    }
            }
        }

        //Store hashed versions of each node, for each new node, check if the node is already in the hash table, if so, match it
        //if the nodes match, swap out the node for an intermediate variable declaration and return, if not, return
        private static List<Node> nodes;
        private static List<Node> subtrees;
        private static void CommonSubtree(Node root)
        {
            for (int i = 0; i < nodes.Count; i++)
                if (root.Operation != NodeOperationType.ConstantDeclaration && root.Operation != NodeOperationType.MatrixDeclaration && root.Operation != NodeOperationType.ConstantMatrixDeclaration && root.Operation != NodeOperationType.IntermediateDeclaration && nodes[i] == root)
                {
                    //Swap out the node for an intermediate variable
                    subtrees.Add(new Node(root));

                    root.IncomingEdges.Clear();
                    root.Operation = NodeOperationType.IntermediateDeclaration;
                    root.OpSpecificValue = null;
                    root.VariableName = /*rNodeIdx +*/ "_subtree_" + (subtrees.Count - 1);

                    return;
                }

            nodes.Add(root);
            for (int i = 0; i < root.IncomingEdges.Count; i++)
            {
                CommonSubtree(root.IncomingEdges[i]);
            }
        }

        static int subtree_base = 0;
        private static void SubtreeReplace(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
            {
                SubtreeReplace(root.IncomingEdges[i]);
            }

            for (int j = 0; j < subtrees.Count; j++)
            {
                if (j == subtree_base)
                    continue;

                if (root == subtrees[j])
                {
                    root.IncomingEdges.Clear();
                    root.Operation = NodeOperationType.IntermediateDeclaration;
                    root.OpSpecificValue = null;
                    root.VariableName = /*rNodeIdx +*/ "_subtree_" + j;
                }
            }
        }

        private static void SplitMultiply(Node root)
        {
            for (int i = 0; i < root.IncomingEdges.Count; i++)
            {
                SplitMultiply(root.IncomingEdges[i]);
            }

            if (root.Operation == NodeOperationType.Multiply)
            {
                for (int i = 0; i < root.IncomingEdges.Count; i++)
                {
                    if (root.IncomingEdges[i].Operation != NodeOperationType.ConstantMatrixDeclaration && root.IncomingEdges[i].Operation != NodeOperationType.MatrixDeclaration && root.IncomingEdges[i].Operation != NodeOperationType.IntermediateDeclaration)
                    {
                        subtrees.Add(new Node(root.IncomingEdges[i]));

                        root.IncomingEdges[i].IncomingEdges.Clear();
                        root.IncomingEdges[i].Operation = NodeOperationType.IntermediateDeclaration;
                        root.IncomingEdges[i].OpSpecificValue = null;
                        root.IncomingEdges[i].VariableName = /*rNodeIdx +*/ "_subtree_" + (subtrees.Count - 1);
                    }
                }

                subtrees.Add(new Node(root));

                root.IncomingEdges.Clear();
                root.Operation = NodeOperationType.IntermediateDeclaration;
                root.OpSpecificValue = null;
                root.VariableName = /*rNodeIdx +*/ "_subtree_" + (subtrees.Count - 1);
            }
        }
        #endregion

        class SingleGraph
        {
            public List<VariableDecl> InputVariables;

            public Node RootTree;
            public List<Node> SubTrees;
            public static Dictionary<string, SubtreeGraph> ReferenceTree;
            public int rNodeIdx;

            static SingleGraph()
            {
                ReferenceTree = new Dictionary<string, SubtreeGraph>();
            }

            public SingleGraph(int rNodeIdx, List<VariableDecl> inputs, Node root, List<Node> subtrees)
            {
                InputVariables = inputs;
                RootTree = root;
                SubTrees = subtrees;
                this.rNodeIdx = rNodeIdx;

                ReferenceTree[rNodeIdx + "_root"] = new SubtreeGraph(rNodeIdx + "_root", root);
                for (int i = 0; i < subtrees.Count; i++)
                {
                    ReferenceTree[/*rNodeIdx +*/ "_subtree_" + i] = new SubtreeGraph("_subtree_" + i, subtrees[i]);
                }

                BuildReferenceTree(rNodeIdx + "_root", root);
                for (int i = 0; i < subtrees.Count; i++)
                {
                    BuildReferenceTree(/*rNodeIdx +*/ "_subtree_" + i, subtrees[i]);
                }

                //Sort ReferenceTree ascending by ReferencedTree count
            }

            public static SubtreeGraph[] GetOrderedSubtreeGraphs()
            {
                var refQueue = ReferenceTree.Values.OrderBy((a) => a.ReferencedTrees.Count).ToList();
                var orderedList = new List<SubtreeGraph>();

                //Find all the pure functions
                for (int i = 0; i < refQueue.Count; i++)
                {
                    if (refQueue[i].ReferencedTrees.Count == 0)
                    {
                        orderedList.Add(refQueue[i]);
                        refQueue.RemoveAt(i);
                        i--;
                    }
                }

                //Now find all the functions that are functions of the currently sorted functions, repeating until there are no more functions to sort
                while (refQueue.Count > 0)
                {
                    for (int i = 0; i < refQueue.Count; i++)
                    {
                        bool allOrdered = true;
                        for (int j = 0; j < refQueue[i].ReferencedTrees.Count; j++)
                        {
                            if (!orderedList.Contains(refQueue[i].ReferencedTrees[j]))
                                allOrdered = false;
                        }

                        if (allOrdered)
                        {
                            orderedList.Add(refQueue[i]);
                            refQueue.RemoveAt(i);
                            i--;
                        }
                    }
                }

                return orderedList.ToArray();
            }

            private void BuildReferenceTree(string parentName, Node node)
            {
                if (node.Operation == NodeOperationType.IntermediateDeclaration && !ReferenceTree[parentName].ReferencedTrees.Contains(ReferenceTree[node.VariableName]))
                {
                    ReferenceTree[node.VariableName].ReferencingTrees.Add(ReferenceTree[parentName]);
                    ReferenceTree[parentName].ReferencedTrees.Add(ReferenceTree[node.VariableName]);
                }

                for (int i = 0; i < node.IncomingEdges.Count; i++)
                {
                    BuildReferenceTree(parentName, node.IncomingEdges[i]);
                }
            }
        }

        class SubtreeGraph
        {
            public string Name { get; private set; }
            public Node Subtree { get; private set; }
            public List<SubtreeGraph> ReferencedTrees { get; private set; }
            public List<SubtreeGraph> ReferencingTrees { get; private set; }

            public SubtreeGraph(string name, Node subtree)
            {
                Name = name;
                Subtree = subtree;
                ReferencedTrees = new List<SubtreeGraph>();
                ReferencingTrees = new List<SubtreeGraph>();
            }

            public static bool operator ==(SubtreeGraph a, SubtreeGraph b)
            {
                return a.Name == b.Name;
            }

            public static bool operator !=(SubtreeGraph a, SubtreeGraph b)
            {
                return a.Name != b.Name;
            }

            public override bool Equals(object obj)
            {
                if (obj is SubtreeGraph)
                {
                    return this == (obj as SubtreeGraph);
                }
                return base.Equals(obj);
            }
        }

        private static SingleGraph BuildSingle(Node root)
        {

            //First gather all variable declarations and verify that they are all uniquely named unless they're the same variable
            List<VariableDecl> variables = new List<VariableDecl>();


            //Simplify the tree, adding intermediate variables for nodes with multiple outputs
            SwapExpr(root);
            ConstExpr(root);
            ConstExpr2(root);
            ConstExpr(root);
            ConstExpr2(root);

            ConstMatExpr(root);
            Console.WriteLine($"{root.Dimension[0]},{root.Dimension[1]}");
            Console.WriteLine(root);
            Console.WriteLine();
            ConstMatExpr2(root);
            ConstMatExpr(root);
            ConstMatExpr2(root);

            //TODO: Figure out how to move subtree identification into also finding common roots, perhaps return root Node, then compute subtrees and common nodes afterwards
            //Simplification: Evaluate all constant expressions, remove all multiplication by zero nodes and then split out common subtrees
            //SplitMultiply(root);
            CommonSubtree(root);
            for (int i = 0; i < subtrees.Count; i++)
            {
                nodes.Clear();
                CommonSubtree(subtrees[i]);
            }

            subtree_base = -1;
            SubtreeReplace(root);
            for (int i = 0; i < subtrees.Count; i++)
            {
                subtree_base = i;
                SubtreeReplace(subtrees[i]);
            }

            /*Console.WriteLine("\n\n\nSimplified:\n");
            Console.WriteLine(root.ToString());

            for (int i = 0; i < subtrees.Count; i++)
            {
                Console.WriteLine();
                Console.Write(rNodeIdx + "_subtree_" + i + " = ");
                Console.WriteLine(subtrees[i]);
            }*/

            return new SingleGraph(rNodeIdx, variables, root, subtrees);
        }
    }
}
