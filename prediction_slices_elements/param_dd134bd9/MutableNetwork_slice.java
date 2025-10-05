// Source-based slice around line 43
// Method: <com.google.common.graph.MutableNetwork: boolean addNode(N)>


  /**
   * Adds {@code node} if it is not already present.
   *
   * <p><b>Nodes must be unique</b>, just as {@code Map} keys must be. They must also be non-null.
   *
   * @return {@code true} if the network was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean addNode(N node);

  /**
   * Adds {@code edge} connecting {@code nodeU} to {@code nodeV}.
   *
   * <p>If the graph is directed, {@code edge} will be directed in this graph; otherwise, it will be
   * undirected.
   *
   * <p><b>{@code edge} must be unique to this graph</b>, just as a {@code Map} key must be. It must
   * also be non-null.
   *
