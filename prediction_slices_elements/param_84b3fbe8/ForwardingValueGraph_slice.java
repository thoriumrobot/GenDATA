// Source-based slice around line 74
// Method: <com.google.common.graph.ForwardingValueGraph: Set predecessors(N)>

    return delegate().incidentEdgeOrder();
  }

  @Override
  public Set<N> adjacentNodes(N node) {
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
  public int degree(N node) {
