// Source-based slice around line 45
// Method: <com.google.common.graph.UndirectedNetworkConnections: UndirectedNetworkConnections ofImmutable(Map)>


  UndirectedNetworkConnections(Map<E, N> incidentEdgeMap) {
    super(incidentEdgeMap);
  }

  static <N, E> UndirectedNetworkConnections<N, E> of() {
    return new UndirectedNetworkConnections<>(HashBiMap.<E, N>create(EXPECTED_DEGREE));
  }

  static <N, E> UndirectedNetworkConnections<N, E> ofImmutable(Map<E, N> incidentEdges) {
    return new UndirectedNetworkConnections<>(ImmutableBiMap.copyOf(incidentEdges));
  }

  @Override
  public Set<N> adjacentNodes() {
    return Collections.unmodifiableSet(((BiMap<E, N>) incidentEdgeMap).values());
  }

  @Override
  public Set<E> edgesConnecting(N node) {
