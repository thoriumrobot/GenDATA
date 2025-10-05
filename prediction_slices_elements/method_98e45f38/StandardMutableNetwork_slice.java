// Source-based slice around line 129
// Method: <com.google.common.graph.StandardMutableNetwork: boolean removeNode(N)>

  @Override
  @CanIgnoreReturnValue
  public boolean addEdge(EndpointPair<N> endpoints, E edge) {
    validateEndpoints(endpoints);
    return addEdge(endpoints.nodeU(), endpoints.nodeV(), edge);
  }

  @Override
  @CanIgnoreReturnValue
  public boolean removeNode(N node) {
    checkNotNull(node, "node");

    NetworkConnections<N, E> connections = nodeConnections.get(node);
    if (connections == null) {
      return false;
    }

    // Since views are returned, we need to copy the edges that will be removed.
    // Thus we avoid modifying the underlying view while iterating over it.
    for (E edge : ImmutableList.copyOf(connections.incidentEdges())) {
