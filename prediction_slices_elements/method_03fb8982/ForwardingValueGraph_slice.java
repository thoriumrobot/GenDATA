// Source-based slice around line 89
// Method: <com.google.common.graph.ForwardingValueGraph: int inDegree(N)>

    return delegate().successors(node);
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
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
