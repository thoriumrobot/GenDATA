// Source-based slice around line 195
// Method: <com.google.common.graph.AbstractNetwork: Optional edgeConnecting(N,N)>

    validateEndpoints(endpoints);
    return edgesConnecting(endpoints.nodeU(), endpoints.nodeV());
  }

  private Predicate<E> connectedPredicate(N nodePresent, N nodeToCheck) {
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
