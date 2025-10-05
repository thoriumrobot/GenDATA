// Source-based slice around line 42
// Method: <com.google.common.graph.MutableGraph: boolean addNode(N)>


  /**
   * Adds {@code node} if it is not already present.
   *
   * <p><b>Nodes must be unique</b>, just as {@code Map} keys must be. They must also be non-null.
   *
   * @return {@code true} if the graph was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean addNode(N node);

  /**
   * Adds an edge connecting {@code nodeU} to {@code nodeV} if one is not already present.
   *
   * <p>If the graph is directed, the resultant edge will be directed; otherwise, it will be
   * undirected.
   *
   * <p>If {@code nodeU} and {@code nodeV} are not already present in this graph, this method will
   * silently {@link #addNode(Object) add} {@code nodeU} and {@code nodeV} to the graph.
   *
