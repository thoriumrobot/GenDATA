// Source-based slice around line 58
// Method: <com.google.common.graph.MutableGraph: boolean putEdge(N,N)>

   *
   * <p>If {@code nodeU} and {@code nodeV} are not already present in this graph, this method will
   * silently {@link #addNode(Object) add} {@code nodeU} and {@code nodeV} to the graph.
   *
   * @return {@code true} if the graph was modified as a result of this call
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsSelfLoops()}
   */
  @CanIgnoreReturnValue
  boolean putEdge(N nodeU, N nodeV);

  /**
   * Adds an edge connecting {@code endpoints} (in the order, if any, specified by {@code
   * endpoints}) if one is not already present.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered and the added edge will be
   * directed; if it is undirected, the added edge will be undirected.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
