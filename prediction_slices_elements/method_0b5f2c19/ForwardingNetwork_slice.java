// Source-based slice around line 155
// Method: <com.google.common.graph.ForwardingNetwork: boolean hasEdgeConnecting(N,N)>

    return delegate().edgeConnectingOrNull(nodeU, nodeV);
  }

  @Override
  public @Nullable E edgeConnectingOrNull(EndpointPair<N> endpoints) {
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
