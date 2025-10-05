// Source-based slice around line 164
// Method: <com.google.common.graph.StandardNetwork: Set outEdges(N)>

    return nodePairInvalidatableSet(connectionsU.edgesConnecting(nodeV), nodeU, nodeV);
  }

  @Override
  public Set<E> inEdges(N node) {
    return nodeInvalidatableSet(checkedConnections(node).inEdges(), node);
  }

  @Override
  public Set<E> outEdges(N node) {
    return nodeInvalidatableSet(checkedConnections(node).outEdges(), node);
  }

  @Override
  public Set<N> predecessors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).predecessors(), node);
  }

  @Override
  public Set<N> successors(N node) {
