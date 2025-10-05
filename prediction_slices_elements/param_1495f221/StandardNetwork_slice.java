// Source-based slice around line 149
// Method: <com.google.common.graph.StandardNetwork: Set edgesConnecting(N,N)>

    return EndpointPair.of(this, nodeU, nodeV);
  }

  @Override
  public Set<N> adjacentNodes(N node) {
    return nodeInvalidatableSet(checkedConnections(node).adjacentNodes(), node);
  }

  @Override
  public Set<E> edgesConnecting(N nodeU, N nodeV) {
    NetworkConnections<N, E> connectionsU = checkedConnections(nodeU);
    if (!allowsSelfLoops && nodeU == nodeV) { // just an optimization, only check reference equality
      return ImmutableSet.of();
    }
    checkArgument(containsNode(nodeV), NODE_NOT_IN_GRAPH, nodeV);
    return nodePairInvalidatableSet(connectionsU.edgesConnecting(nodeV), nodeU, nodeV);
  }

  @Override
  public Set<E> inEdges(N node) {
