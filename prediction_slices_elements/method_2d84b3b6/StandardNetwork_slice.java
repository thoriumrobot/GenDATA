// Source-based slice around line 136
// Method: <com.google.common.graph.StandardNetwork: EndpointPair incidentNodes(E)>

    return edgeOrder;
  }

  @Override
  public Set<E> incidentEdges(N node) {
    return nodeInvalidatableSet(checkedConnections(node).incidentEdges(), node);
  }

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
