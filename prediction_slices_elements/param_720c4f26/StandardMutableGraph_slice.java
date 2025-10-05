// Source-based slice around line 45
// Method: <com.google.common.graph.StandardMutableGraph: boolean addNode(N)>

    this.backingValueGraph = new StandardMutableValueGraph<>(builder);
  }

  @Override
  BaseGraph<N> delegate() {
    return backingValueGraph;
  }

  @Override
  public boolean addNode(N node) {
    return backingValueGraph.addNode(node);
  }

  @Override
  public boolean putEdge(N nodeU, N nodeV) {
    return backingValueGraph.putEdgeValue(nodeU, nodeV, Presence.EDGE_EXISTS) == null;
  }

  @Override
  public boolean putEdge(EndpointPair<N> endpoints) {
