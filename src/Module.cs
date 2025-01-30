using System;
using System.Collections.Generic;
using System.Reflection.Metadata.Ecma335;
using Micrograd;

public abstract class Module
{
    public virtual void ZeroGrad()
    {
        foreach(var p in Parameters())
        {
            p.Grad = 0;
        }
    }

    public virtual List<Value> Parameters() => new List<Value>();

}