// Source-based slice around line 148
// Method: <com.google.common.graph.StandardMutableNetwork: boolean removeEdge(E)>

    for (E edge : ImmutableList.copyOf(connections.incidentEdges())) {
      removeEdge(edge);
    }
    nodeConnections.remove(node);
    return true;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean removeEdge(E edge) {
    checkNotNull(edge, "edge");

    N nodeU = edgeToReferenceNode.get(edge);
    if (nodeU == null) {
      return false;
    }

    // requireNonNull is safe because of the edgeToReferenceNode check above.
    NetworkConnections<N, E> connectionsU = requireNonNull(nodeConnections.get(nodeU));
    N nodeV = connectionsU.adjacentNode(edge);
