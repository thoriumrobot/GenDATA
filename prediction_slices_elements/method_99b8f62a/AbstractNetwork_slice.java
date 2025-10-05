// Source-based slice around line 232
// Method: <com.google.common.graph.AbstractNetwork: boolean hasEdgeConnecting(EndpointPair)>


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
    }
    return hasEdgeConnecting(endpoints.nodeU(), endpoints.nodeV());
  }

  /**
   * Throws an IllegalArgumentException if the ordering of {@code endpoints} is not compatible with
   * the directionality of this graph.
