// Source-based slice around line 198
// Method: <com.google.common.graph.BaseGraph: int outDegree(N)>


  /**
   * Returns the count of {@code node}'s outgoing edges (equal to {@code successors(node).size()})
   * in a directed graph. In an undirected graph, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  int outDegree(N node);

  /**
   * Returns true if there is an edge that directly connects {@code nodeU} to {@code nodeV}. This is
   * equivalent to {@code nodes().contains(nodeU) && successors(nodeU).contains(nodeV)}.
   *
   * <p>In an undirected graph, this is equal to {@code hasEdgeConnecting(nodeV, nodeU)}.
   *
   * @since 23.0
   */
  boolean hasEdgeConnecting(N nodeU, N nodeV);
