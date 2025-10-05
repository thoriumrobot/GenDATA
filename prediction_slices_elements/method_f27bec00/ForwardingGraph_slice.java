// Source-based slice around line 86
// Method: <com.google.common.graph.ForwardingGraph: int degree(N)>

    return delegate().successors(node);
  }

  @Override
  public Set<EndpointPair<N>> incidentEdges(N node) {
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
