// Source-based slice around line 185
// Method: <com.google.common.graph.AbstractNetwork: Set edgesConnecting(EndpointPair)>

    return nodePairInvalidatableSet(
        outEdgesU.size() <= inEdgesV.size()
            ? unmodifiableSet(Sets.filter(outEdgesU, connectedPredicate(nodeU, nodeV)))
            : unmodifiableSet(Sets.filter(inEdgesV, connectedPredicate(nodeV, nodeU))),
        nodeU,
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
