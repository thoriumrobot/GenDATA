// Source-based slice around line 204
// Method: <com.google.common.graph.ValueGraph: Set predecessors(N)>

   * <p>In an undirected graph, this is equivalent to {@link #adjacentNodes(Object)}.
   *
   * <p>If {@code node} is removed from the graph after this method is called, the {@code Set}
   * returned by this method will be invalidated, and will throw {@code IllegalStateException} if it
   * is accessed in any way.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  @Override
  Set<N> predecessors(N node);

  /**
   * Returns a live view of all nodes in this graph adjacent to {@code node} which can be reached by
   * traversing {@code node}'s outgoing edges in the direction (if any) of the edge.
   *
   * <p>In an undirected graph, this is equivalent to {@link #adjacentNodes(Object)}.
   *
   * <p>This is <i>not</i> the same as "all nodes reachable from {@code node} by following outgoing
   * edges". For that functionality, see {@link Graphs#reachableNodes(Graph, Object)}.
   *
