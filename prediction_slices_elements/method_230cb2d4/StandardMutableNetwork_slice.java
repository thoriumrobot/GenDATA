// Source-based slice around line 70
// Method: <com.google.common.graph.StandardMutableNetwork: NetworkConnections addNodeInternal(N)>

    return true;
  }

  /**
   * Adds {@code node} to the graph and returns the associated {@link NetworkConnections}.
   *
   * @throws IllegalStateException if {@code node} is already present
   */
  @CanIgnoreReturnValue
  private NetworkConnections<N, E> addNodeInternal(N node) {
    NetworkConnections<N, E> connections = newConnections();
    checkState(nodeConnections.put(node, connections) == null);
    return connections;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addEdge(N nodeU, N nodeV, E edge) {
    checkNotNull(nodeU, "nodeU");
    checkNotNull(nodeV, "nodeV");
