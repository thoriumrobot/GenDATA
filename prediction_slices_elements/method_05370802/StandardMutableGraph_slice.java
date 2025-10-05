// Source-based slice around line 66
// Method: <com.google.common.graph.StandardMutableGraph: boolean removeEdge(N,N)>

    return putEdge(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public boolean removeNode(N node) {
    return backingValueGraph.removeNode(node);
  }

  @Override
  public boolean removeEdge(N nodeU, N nodeV) {
    return backingValueGraph.removeEdge(nodeU, nodeV) != null;
  }

  @Override
  public boolean removeEdge(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return removeEdge(endpoints.nodeU(), endpoints.nodeV());
  }
}
