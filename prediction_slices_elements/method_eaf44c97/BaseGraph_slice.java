// Source-based slice around line 115
// Method: <com.google.common.graph.BaseGraph: Set predecessors(N)>

   *       involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the graph after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
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
