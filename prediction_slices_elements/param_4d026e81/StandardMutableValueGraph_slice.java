// Source-based slice around line 79
// Method: <com.google.common.graph.StandardMutableValueGraph: GraphConnections addNodeInternal(N)>

    return true;
  }

  /**
   * Adds {@code node} to the graph and returns the associated {@link GraphConnections}.
   *
   * @throws IllegalStateException if {@code node} is already present
   */
  @CanIgnoreReturnValue
  private GraphConnections<N, V> addNodeInternal(N node) {
    GraphConnections<N, V> connections = newConnections();
    checkState(nodeConnections.put(node, connections) == null);
    return connections;
  }

  @Override
  @CanIgnoreReturnValue
  public @Nullable V putEdgeValue(N nodeU, N nodeV, V value) {
    checkNotNull(nodeU, "nodeU");
    checkNotNull(nodeV, "nodeV");
