// Source-based slice around line 109
// Method: <com.google.common.graph.StandardValueGraph: Set predecessors(N)>

    return nodeOrder;
  }

  @Override
  public Set<N> adjacentNodes(N node) {
    return nodeInvalidatableSet(checkedConnections(node).adjacentNodes(), node);
  }

  @Override
  public Set<N> predecessors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).predecessors(), node);
  }

  @Override
  public Set<N> successors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).successors(), node);
  }

  @Override
  public Set<EndpointPair<N>> incidentEdges(N node) {
