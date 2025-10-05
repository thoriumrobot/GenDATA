// Source-based slice around line 63
// Method: <com.google.common.graph.DirectedNetworkConnections: Set edgesConnecting(N)>

    return Collections.unmodifiableSet(((BiMap<E, N>) inEdgeMap).values());
  }

  @Override
  public Set<N> successors() {
    return Collections.unmodifiableSet(((BiMap<E, N>) outEdgeMap).values());
  }

  @Override
  public Set<E> edgesConnecting(N node) {
    return new EdgesConnecting<>(((BiMap<E, N>) outEdgeMap).inverse(), node);
  }
}
