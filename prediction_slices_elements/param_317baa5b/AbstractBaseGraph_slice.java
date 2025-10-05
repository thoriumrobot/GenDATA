// Source-based slice around line 165
// Method: <com.google.common.graph.AbstractBaseGraph: boolean hasEdgeConnecting(EndpointPair)>


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
    N nodeU = endpoints.nodeU();
    N nodeV = endpoints.nodeV();
    return nodes().contains(nodeU) && successors(nodeU).contains(nodeV);
  }

  /**
