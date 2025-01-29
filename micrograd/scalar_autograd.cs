using System;
using System.Collections.Generic;

class Value{
    public double Data{get; private set; }
    public double Grad {get;set;}
    private HashSet<Value> _prev;
    private string _op;
    private Action _backward;

    public Value(double data , HashSet<Value> children = null , string op =""){
        Data = data;
        Grad = 0;
        _prev = children ?? new HashSet<Value>();
        _backward =() =>{ };
        _op = op;
    }

    public static Value operator + (Value a, Value b){
        b= b ?? new Value(0);
        var outValue = new Value(a.Data + b.Data , new HashSet<Value>{a,b},"+");
        return outValue;
    }

    public static Value operator * (Value a, Value b){
        b= b ?? new Value(1);
        var outValue = new Value(a.Data * b.Data , new HashSet<Value>{a,b},"*");
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

        return outValue;
    }
}