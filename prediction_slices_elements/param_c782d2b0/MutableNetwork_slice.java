// Source-based slice around line 67
// Method: <com.google.common.graph.MutableNetwork: boolean addEdge(N,N,E)>

   * this network {@link #isDirected()}, else in any order), then this method will have no effect.
   *
   * @return {@code true} if the network was modified as a result of this call
   * @throws IllegalArgumentException if {@code edge} already exists in the graph and does not
   *     connect {@code nodeU} to {@code nodeV}
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsParallelEdges()} or {@link #allowsSelfLoops()}
   */
  @CanIgnoreReturnValue
  boolean addEdge(N nodeU, N nodeV, E edge);

  /**
   * Adds {@code edge} connecting {@code endpoints}. In an undirected network, {@code edge} will
   * also connect {@code nodeV} to {@code nodeU}.
   *
   * <p>If this graph is directed, {@code edge} will be directed in this graph; if it is undirected,
   * {@code edge} will be undirected in this graph.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
