// Source-based slice around line 158
// Method: <com.google.common.graph.AbstractBaseGraph: boolean hasEdgeConnecting(N,N)>

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

  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
    checkNotNull(endpoints);
    if (!isOrderingCompatible(endpoints)) {
      return false;
