// Source-based slice around line 71
// Method: <com.google.common.graph.StandardMutableGraph: boolean removeEdge(EndpointPair)>

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
