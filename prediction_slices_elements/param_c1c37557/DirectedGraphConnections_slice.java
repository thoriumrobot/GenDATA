// Source-based slice around line 179
// Method: <com.google.common.graph.DirectedGraphConnections: DirectedGraphConnections ofImmutable(N,Iterable,Function)>


    return new DirectedGraphConnections<>(
        /* adjacentNodeValues= */ new HashMap<N, Object>(initialCapacity, INNER_LOAD_FACTOR),
        orderedNodeConnections,
        /* predecessorCount= */ 0,
        /* successorCount= */ 0);
  }

  static <N, V> DirectedGraphConnections<N, V> ofImmutable(
      N thisNode, Iterable<EndpointPair<N>> incidentEdges, Function<N, V> successorNodeToValueFn) {
    checkNotNull(thisNode);
    checkNotNull(successorNodeToValueFn);

    Map<N, Object> adjacentNodeValues = new HashMap<>();
    ImmutableList.Builder<NodeConnection<N>> orderedNodeConnectionsBuilder =
        ImmutableList.builder();
    int predecessorCount = 0;
    int successorCount = 0;

    for (EndpointPair<N> incidentEdge : incidentEdges) {
