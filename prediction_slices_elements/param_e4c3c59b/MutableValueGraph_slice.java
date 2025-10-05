// Source-based slice around line 63
// Method: <com.google.common.graph.MutableValueGraph: V putEdgeValue(N,N,V)>

   * <p>If {@code nodeU} and {@code nodeV} are not already present in this graph, this method will
   * silently {@link #addNode(Object) add} {@code nodeU} and {@code nodeV} to the graph.
   *
   * @return the value previously associated with the edge connecting {@code nodeU} to {@code
   *     nodeV}, or null if there was no such edge.
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsSelfLoops()}
   */
  @CanIgnoreReturnValue
  @Nullable V putEdgeValue(N nodeU, N nodeV, V value);

  /**
   * Adds an edge connecting {@code endpoints} if one is not already present, and sets a value for
   * that edge to {@code value} (overwriting the existing value, if any).
   *
   * <p>If the graph is directed, the resultant edge will be directed; otherwise, it will be
   * undirected.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
