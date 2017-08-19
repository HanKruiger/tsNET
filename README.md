# tsNET

Graph Layouts by t-SNE

```
usage: tsnet.py [-h] [--star] [--perplexity PERPLEXITY]
                [--learning_rate LEARNING_RATE] [--output OUTPUT]
                input_graph

Read a graph, and produce a layout with tsNET(*).

positional arguments:
  input_graph

optional arguments:
  -h, --help            show this help message and exit
  --star                Use the tsNET* scheme. (Requires PivotMDS layout in
                        ./pivotmds_layouts/ as initialization.) Note: Use
                        higher learning rates for larger graphs, for faster
                        convergence.
  --perplexity PERPLEXITY, -p PERPLEXITY
                        Perplexity parameter.
  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate (hyper)parameter for optimization.
  --output OUTPUT, -o OUTPUT
                        Save layout to the specified file.
```

# Dependencies

* `python3`
* [`numpy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`graph-tool`](https://graph-tool.skewed.de/)
* [`theano`](http://deeplearning.net/software/theano/)
* [`scikit-learn`](http://scikit-learn.org/stable/)
