// Source-based slice around line 106
// Method: <com.google.common.graph.ForwardingGraph: boolean hasEdgeConnecting(EndpointPair)>

    return delegate().outDegree(node);
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
