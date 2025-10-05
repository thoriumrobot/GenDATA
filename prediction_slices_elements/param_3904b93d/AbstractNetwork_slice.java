// Source-based slice around line 200
// Method: <com.google.common.graph.AbstractNetwork: Optional edgeConnecting(EndpointPair)>

    return edge -> incidentNodes(edge).adjacentNode(nodePresent).equals(nodeToCheck);
  }

  @Override
  public Optional<E> edgeConnecting(N nodeU, N nodeV) {
    return Optional.ofNullable(edgeConnectingOrNull(nodeU, nodeV));
  }

  @Override
  public Optional<E> edgeConnecting(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return edgeConnecting(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public @Nullable E edgeConnectingOrNull(N nodeU, N nodeV) {
    Set<E> edgesConnecting = edgesConnecting(nodeU, nodeV);
    switch (edgesConnecting.size()) {
      case 0:
        return null;
