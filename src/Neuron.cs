using System;
using System.Collections.Generic;
using System.Linq;
using Micrograd;

public class Neuron:Module
{
    private List<Value> w;
    private Value b;
    private bool nonlin;
    private static Random rand = new Random();


    public Neuron(int nin , bool nonlin = true)
    {
        w = Enumerable.Range(0, nin).Select(_ => new Value(rand.NextDouble() * 2 - 1)).ToList();
        b = new Value(0);
        this.nonlin = nonlin;
    }

    public Value Call(List<Value> x)
    {
        var act = w.Zip(x, (wi, xi) => wi * xi).Aggregate(b, (sum, term) => sum + term);
        return nonlin ? act.Tanh() : act;
    }
    
    public override List<Value> Parameters()=> w.Append(b).ToList();
    public override string ToString() => $"{(nonlin ? "Tanh" : "Linear")}Neuron({w.Count})";
}