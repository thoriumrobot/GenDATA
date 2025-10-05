// Source-based slice around line 160
// Method: <com.google.common.graph.ForwardingNetwork: boolean hasEdgeConnecting(EndpointPair)>

    return delegate().edgeConnectingOrNull(endpoints);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
    return delegate().hasEdgeConnecting(nodeU, nodeV);
  }

  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
    return delegate().hasEdgeConnecting(endpoints);
  }
}
