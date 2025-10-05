// Source-based slice around line 76
// Method: <com.google.common.graph.ForwardingGraph: Set successors(N)>

    return delegate().adjacentNodes(node);
  }

  @Override
  public Set<N> predecessors(N node) {
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
