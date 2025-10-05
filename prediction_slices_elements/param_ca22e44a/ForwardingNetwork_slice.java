// Source-based slice around line 115
// Method: <com.google.common.graph.ForwardingNetwork: int inDegree(N)>

    return delegate().adjacentEdges(edge);
  }

  @Override
  public int degree(N node) {
    return delegate().degree(node);
  }

  @Override
  public int inDegree(N node) {
    return delegate().inDegree(node);
  }

  @Override
  public int outDegree(N node) {
    return delegate().outDegree(node);
  }

  @Override
  public Set<E> edgesConnecting(N nodeU, N nodeV) {
