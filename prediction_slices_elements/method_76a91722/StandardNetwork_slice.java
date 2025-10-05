// Source-based slice around line 159
// Method: <com.google.common.graph.StandardNetwork: Set inEdges(N)>

    NetworkConnections<N, E> connectionsU = checkedConnections(nodeU);
    if (!allowsSelfLoops && nodeU == nodeV) { // just an optimization, only check reference equality
      return ImmutableSet.of();
    }
    checkArgument(containsNode(nodeV), NODE_NOT_IN_GRAPH, nodeV);
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
