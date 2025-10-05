// Source-based slice around line 91
// Method: <com.google.common.graph.ForwardingGraph: int inDegree(N)>

    return delegate().incidentEdges(node);
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
