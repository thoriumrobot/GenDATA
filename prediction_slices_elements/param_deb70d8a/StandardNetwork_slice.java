// Source-based slice around line 144
// Method: <com.google.common.graph.StandardNetwork: Set adjacentNodes(N)>

  @Override
  public EndpointPair<N> incidentNodes(E edge) {
    N nodeU = checkedReferenceNode(edge);
    // requireNonNull is safe because checkedReferenceNode made sure the edge is in the network.
    N nodeV = requireNonNull(nodeConnections.get(nodeU)).adjacentNode(edge);
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
