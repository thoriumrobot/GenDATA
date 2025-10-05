// Source-based slice around line 62
// Method: com.google.common.graph.StandardNetwork.edgeToReferenceNode

  private final boolean allowsParallelEdges;
  private final boolean allowsSelfLoops;
  private final ElementOrder<N> nodeOrder;
  private final ElementOrder<E> edgeOrder;

  final MapIteratorCache<N, NetworkConnections<N, E>> nodeConnections;

  // We could make this a Map<E, EndpointPair<N>>. It would make incidentNodes(edge) slightly
  // faster, but also make Networks consume 5 to 20+% (increasing with average degree) more memory.
  final MapIteratorCache<E, N> edgeToReferenceNode; // referenceNode == source if directed

  /** Constructs a graph with the properties specified in {@code builder}. */
  StandardNetwork(NetworkBuilder<? super N, ? super E> builder) {
    this(
        builder,
        builder.nodeOrder.<N, NetworkConnections<N, E>>createMap(
            builder.expectedNodeCount.or(DEFAULT_NODE_COUNT)),
        builder.edgeOrder.<E, N>createMap(builder.expectedEdgeCount.or(DEFAULT_EDGE_COUNT)));
  }

