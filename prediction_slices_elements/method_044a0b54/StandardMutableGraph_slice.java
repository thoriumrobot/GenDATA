// Source-based slice around line 55
// Method: <com.google.common.graph.StandardMutableGraph: boolean putEdge(EndpointPair)>

    return backingValueGraph.addNode(node);
  }

  @Override
  public boolean putEdge(N nodeU, N nodeV) {
    return backingValueGraph.putEdgeValue(nodeU, nodeV, Presence.EDGE_EXISTS) == null;
  }

  @Override
  public boolean putEdge(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return putEdge(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public boolean removeNode(N node) {
    return backingValueGraph.removeNode(node);
  }

  @Override
