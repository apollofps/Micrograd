using System;
using System.Collections.Generic;
namespace Micrograd
{
    class Value
    {
        public double Data { get; private set; }
        public double Grad { get; set; }
        private HashSet<Value> _prev;
        private string _op;
        private Action _backwardPropogation;

        public Value(double data, HashSet<Value> children = null, string op = "")
        {
            Data = data;
            Grad = 0;
            _prev = children ?? new HashSet<Value>();
            _backwardPropogation = () => { };
            _op = op;
        }

        public static Value operator +(Value a, Value b)
        {
            b = b ?? new Value(0);
            var outValue = new Value(a.Data + b.Data, new HashSet<Value> { a, b }, "+");

            outValue._backwardPropogation = () =>
            {
                a.Grad += outValue.Grad;
                b.Grad += outValue.Grad;
            };

            return outValue;
        }

        public static Value operator *(Value a, Value b)
        {
            b = b ?? new Value(1);
            var outValue = new Value(a.Data * b.Data, new HashSet<Value> { a, b }, "*");
            // chanin value rule 
            outValue._backwardPropogation = () =>
            {
                a.Grad += b.Data * outValue.Grad;
                b.Grad += a.Data * outValue.Grad;
            };


            return outValue;
        }

        public static Value operator -(Value a, Value b)
        {
            return a + (-b);
        }

        public static Value operator -(Value a)
        {
            return a * new Value(-1);
        }



        public Value Pow(double exponent)
        {
            var outValue = new Value(Math.Pow(Data, exponent), new HashSet<Value> { this }, $"**{exponent}");

            outValue._backwardPropogation = () =>
            {
                Grad += (exponent * Math.Pow(Data, exponent - 1)) * outValue.Grad;
            };

            return outValue;
        }


        public Value Exp()
        {
            double x = Data;
            var outValue = new Value(Math.Exp(x), new HashSet<Value> { this }, "exp");

            outValue._backwardPropogation = () =>
            {
                Grad += outValue.Data * outValue.Grad;
            };

            return outValue;
        }
        // public Value Tanh()
        // {
        //     double tanhValue = Math.Tanh(Data);
        //     var outValue = new Value(tanhValue, new HashSet<Value> { this }, "tanh");
        //     return outValue;
        // }



        // tanh(x)= (e^2x -1) /(e^2x + 1)
        public Value Tanh()
        {
            double exp2x = Math.Exp(2 * Data);
            double tanhValue = (exp2x - 1) / (exp2x + 1);
            var outValue = new Value(tanhValue, new HashSet<Value> { this }, "tanh");
            //derivative of tan(x)
            outValue._backwardPropogation = () =>
            {
                Grad += (1 - tanhValue * tanhValue) * outValue.Grad;
            };


            return outValue;
        }

        public void Backward()
        {
            var topologicalGraph = new List<Value>();
            var visitedNodes = new HashSet<Value>();

            void buildTopologicalGraph(Value v)
            {
                if (!visitedNodes.Contains(v))
                {
                    visitedNodes.Add(v);
                    foreach (var child in v._prev)
                        buildTopologicalGraph(child);
                    topologicalGraph.Add(v);
                }
            }

            buildTopologicalGraph(this);

            Grad = 1;
            for (int i = topologicalGraph.Count - 1; i >= 0; i--)
            {
                topologicalGraph[i]._backwardPropogation();
            }
        }

        public override string ToString() => $"Value(data={Data}, grad={Grad})";
    }
}

