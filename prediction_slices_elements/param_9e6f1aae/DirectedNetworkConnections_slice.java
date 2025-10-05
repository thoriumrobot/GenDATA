// Source-based slice around line 47
// Method: <com.google.common.graph.DirectedNetworkConnections: DirectedNetworkConnections ofImmutable(Map,Map,int)>

    super(inEdgeMap, outEdgeMap, selfLoopCount);
  }

  static <N, E> DirectedNetworkConnections<N, E> of() {
    return new DirectedNetworkConnections<>(
        HashBiMap.<E, N>create(EXPECTED_DEGREE), HashBiMap.<E, N>create(EXPECTED_DEGREE), 0);
  }

  static <N, E> DirectedNetworkConnections<N, E> ofImmutable(
      Map<E, N> inEdges, Map<E, N> outEdges, int selfLoopCount) {
    return new DirectedNetworkConnections<>(
        ImmutableBiMap.copyOf(inEdges), ImmutableBiMap.copyOf(outEdges), selfLoopCount);
  }

  @Override
  public Set<N> predecessors() {
    return Collections.unmodifiableSet(((BiMap<E, N>) inEdgeMap).values());
  }

  @Override
