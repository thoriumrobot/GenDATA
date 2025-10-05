// Source-based slice around line 190
// Method: <com.google.common.graph.AbstractNetwork: Predicate connectedPredicate(N,N)>

        nodeV);
  }

  @Override
  public Set<E> edgesConnecting(EndpointPair<N> endpoints) {
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
