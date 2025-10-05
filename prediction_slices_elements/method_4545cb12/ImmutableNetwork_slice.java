// Source-based slice around line 82
// Method: <com.google.common.graph.ImmutableNetwork: Map getNodeConnections(Network)>

  public static <N, E> ImmutableNetwork<N, E> copyOf(ImmutableNetwork<N, E> network) {
    return checkNotNull(network);
  }

  @Override
  public ImmutableGraph<N> asGraph() {
    return new ImmutableGraph<>(super.asGraph()); // safe because the view is effectively immutable
  }

  private static <N, E> Map<N, NetworkConnections<N, E>> getNodeConnections(Network<N, E> network) {
    // ImmutableMap.Builder maintains the order of the elements as inserted, so the map will have
    // whatever ordering the network's nodes do, so ImmutableSortedMap is unnecessary even if the
    // input nodes are sorted.
    ImmutableMap.Builder<N, NetworkConnections<N, E>> nodeConnections = ImmutableMap.builder();
    for (N node : network.nodes()) {
      nodeConnections.put(node, connectionsOf(network, node));
    }
    return nodeConnections.buildOrThrow();
  }

