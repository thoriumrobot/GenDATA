// Source-based slice around line 43
// Method: <com.google.common.graph.MutableValueGraph: boolean addNode(N)>


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
   * Adds an edge connecting {@code nodeU} to {@code nodeV} if one is not already present, and sets
   * a value for that edge to {@code value} (overwriting the existing value, if any).
   *
   * <p>If the graph is directed, the resultant edge will be directed; otherwise, it will be
   * undirected.
   *
   * <p>Values do not have to be unique. However, values must be non-null.
   *
