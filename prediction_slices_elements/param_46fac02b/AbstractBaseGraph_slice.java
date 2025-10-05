// Source-based slice around line 153
// Method: <com.google.common.graph.AbstractBaseGraph: int outDegree(N)>

    }
  }

  @Override
  public int inDegree(N node) {
    return isDirected() ? predecessors(node).size() : degree(node);
  }

  @Override
  public int outDegree(N node) {
    return isDirected() ? successors(node).size() : degree(node);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
    checkNotNull(nodeU);
    checkNotNull(nodeV);
    return nodes().contains(nodeU) && successors(nodeU).contains(nodeV);
  }

