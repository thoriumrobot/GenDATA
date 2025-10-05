// Source-based slice around line 189
// Method: <com.google.common.graph.ValueGraph: Set adjacentNodes(N)>

   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the graph after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  @Override
  Set<N> adjacentNodes(N node);

  /**
   * Returns a live view of all nodes in this graph adjacent to {@code node} which can be reached by
   * traversing {@code node}'s incoming edges <i>against</i> the direction (if any) of the edge.
   *
   * <p>In an undirected graph, this is equivalent to {@link #adjacentNodes(Object)}.
   *
   * <p>If {@code node} is removed from the graph after this method is called, the {@code Set}
   * returned by this method will be invalidated, and will throw {@code IllegalStateException} if it
   * is accessed in any way.
