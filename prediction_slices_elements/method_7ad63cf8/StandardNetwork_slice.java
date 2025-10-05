// Source-based slice around line 126
// Method: <com.google.common.graph.StandardNetwork: ElementOrder edgeOrder()>

    return allowsSelfLoops;
  }

  @Override
  public ElementOrder<N> nodeOrder() {
    return nodeOrder;
  }

  @Override
  public ElementOrder<E> edgeOrder() {
    return edgeOrder;
  }

  @Override
  public Set<E> incidentEdges(N node) {
    return nodeInvalidatableSet(checkedConnections(node).incidentEdges(), node);
  }

  @Override
  public EndpointPair<N> incidentNodes(E edge) {
