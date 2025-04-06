import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def _(torch):
    x = torch.arange(12, dtype=torch.float32)
    x
    return (x,)


@app.cell
def _(x):
    x.numel()
    return


@app.cell
def _(x):
    x.shape
    return


@app.cell
def _(x):
    X = x.reshape(3,4)
    X
    return (X,)


@app.cell
def _(torch):
    torch.zeros((2,4,5))
    return


@app.cell
def _(torch):
    torch.ones((2,5,6))
    return


@app.cell
def _(X):
    X[-1]
    return


@app.cell
def _(X):
    X[1,2]
    return


@app.cell
def _(X):
    X[:2, :] = -1
    X
    return


@app.cell
def _(torch):
    help(torch.exp)
    return


@app.cell
def _(X, torch):
    torch.exp(X)
    return


@app.cell
def _(torch):
    a = torch.randn((2,3,4))
    a
    return (a,)


@app.cell
def _(torch):
    b = torch.randn((3,4,5))
    b
    return (b,)


@app.cell
def _(a, b):
    c = a * b
    return (c,)


@app.cell
def _(a, b):
    # reshaping the tensors to apply broadcasting operation
    a.shape, b.shape
    return


@app.cell
def _(a, b):
    d = b.reshape(1, 3, -1)
    print(d.shape)

    j = a.reshape(1,3,-1)
    print(j.shape)
    return d, j


@app.cell
def _(torch):
    k = torch.randn((1,3,5))
    l = torch.randn((1,1,5))
    e = k * l
    e
    return e, k, l


@app.cell
def _(mo):
    mo.md(
        """
        Broadcasting works only on 2 cases. <br>
        1. Look at the dimensions pairwise and compare them from right to left <br>
        2. Take 2 tensors (1,3,5) and (1,1,5) <br>
        3. Here 1->1 matched, 3-> 1 not matched but one of them is 1 so its valid and 5->5 match, hence these two tensors can be broadcasted <br>
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
