using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd
{
    public class Layer : Module
    {
        private List<Neuron> neurons;

        public Layer(int nin , int nout , bool nonlin = true)
        {
            neurons = Enumerable.Range(0, nout).Select(_ => new Neuron(nin, nonlin)).ToList();
        }

        public List<Value> Call(List<Value> x)
        {
            var outValues = neurons.Select(n => n.Call(x)).ToList();
            return outValues.Count == 1 ? new List<Value> { outValues[0] } : outValues;
        }
        //list all neurons and its parameters
        public override List<Value> Parameters() => neurons.SelectMany(n => n.Parameters()).ToList();

        public override string ToString() => $"Layer of [{string.Join(", ", neurons)}]";
    } 
}