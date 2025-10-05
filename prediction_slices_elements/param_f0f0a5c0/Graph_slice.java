// Source-based slice around line 277
// Method: <com.google.common.graph.Graph: int inDegree(N)>

  /**
   * Returns the count of {@code node}'s incoming edges (equal to {@code predecessors(node).size()})
   * in a directed graph. In an undirected graph, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  @Override
  int inDegree(N node);

  /**
   * Returns the count of {@code node}'s outgoing edges (equal to {@code successors(node).size()})
   * in a directed graph. In an undirected graph, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  @Override
