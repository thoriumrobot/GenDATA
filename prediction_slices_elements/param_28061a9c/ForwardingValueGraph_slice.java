// Source-based slice around line 79
// Method: <com.google.common.graph.ForwardingValueGraph: Set successors(N)>

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
    return delegate().degree(node);
  }

  @Override
  public int inDegree(N node) {
