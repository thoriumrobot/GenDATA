// Source-based slice around line 173
// Method: <com.google.common.graph.AbstractNetwork: Set edgesConnecting(N,N)>

  public Set<E> adjacentEdges(E edge) {
    EndpointPair<N> endpointPair = incidentNodes(edge); // Verifies that edge is in this network.
    Set<E> endpointPairIncidentEdges =
        Sets.union(incidentEdges(endpointPair.nodeU()), incidentEdges(endpointPair.nodeV()));
    return edgeInvalidatableSet(
        Sets.difference(endpointPairIncidentEdges, ImmutableSet.of(edge)), edge);
  }

  @Override
  public Set<E> edgesConnecting(N nodeU, N nodeV) {
    Set<E> outEdgesU = outEdges(nodeU);
    Set<E> inEdgesV = inEdges(nodeV);
    return nodePairInvalidatableSet(
        outEdgesU.size() <= inEdgesV.size()
            ? unmodifiableSet(Sets.filter(outEdgesU, connectedPredicate(nodeU, nodeV)))
            : unmodifiableSet(Sets.filter(inEdgesV, connectedPredicate(nodeV, nodeU))),
        nodeU,
        nodeV);
  }

