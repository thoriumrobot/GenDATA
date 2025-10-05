// Source-based slice around line 58
// Method: <com.google.common.graph.DirectedNetworkConnections: Set successors()>

        ImmutableBiMap.copyOf(inEdges), ImmutableBiMap.copyOf(outEdges), selfLoopCount);
  }

  @Override
  public Set<N> predecessors() {
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
