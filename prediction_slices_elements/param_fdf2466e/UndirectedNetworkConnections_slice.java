// Source-based slice around line 55
// Method: <com.google.common.graph.UndirectedNetworkConnections: Set edgesConnecting(N)>

    return new UndirectedNetworkConnections<>(ImmutableBiMap.copyOf(incidentEdges));
  }

  @Override
  public Set<N> adjacentNodes() {
    return Collections.unmodifiableSet(((BiMap<E, N>) incidentEdgeMap).values());
  }

  @Override
  public Set<E> edgesConnecting(N node) {
    return new EdgesConnecting<>(((BiMap<E, N>) incidentEdgeMap).inverse(), node);
  }
}
