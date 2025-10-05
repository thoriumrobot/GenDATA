// Source-based slice around line 81
// Method: <com.google.common.graph.ForwardingGraph: Set incidentEdges(N)>

    return delegate().predecessors(node);
  }

  @Override
  public Set<N> successors(N node) {
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
