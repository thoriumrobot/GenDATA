// Source-based slice around line 321
// Method: <com.google.common.graph.Network: int inDegree(N)>


  /**
   * Returns the count of {@code node}'s {@link #inEdges(Object) incoming edges} in a directed
   * network. In an undirected network, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  int inDegree(N node);

  /**
   * Returns the count of {@code node}'s {@link #outEdges(Object) outgoing edges} in a directed
   * network. In an undirected network, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  int outDegree(N node);
