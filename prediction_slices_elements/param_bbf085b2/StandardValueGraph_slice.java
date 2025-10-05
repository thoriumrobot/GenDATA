// Source-based slice around line 104
// Method: <com.google.common.graph.StandardValueGraph: Set adjacentNodes(N)>

    return allowsSelfLoops;
  }

  @Override
  public ElementOrder<N> nodeOrder() {
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
