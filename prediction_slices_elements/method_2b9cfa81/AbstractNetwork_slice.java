// Source-based slice around line 164
// Method: <com.google.common.graph.AbstractNetwork: Set adjacentEdges(E)>

    return isDirected() ? inEdges(node).size() : degree(node);
  }

  @Override
  public int outDegree(N node) {
    return isDirected() ? outEdges(node).size() : degree(node);
  }

  @Override
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
