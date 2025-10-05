// Source-based slice around line 250
// Method: <com.google.common.graph.Graph: Set incidentEdges(N)>

   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the graph after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   * @since 24.0
   */
  @Override
  Set<EndpointPair<N>> incidentEdges(N node);

  /**
   * Returns the count of {@code node}'s incident edges, counting self-loops twice (equivalently,
   * the number of times an edge touches {@code node}).
   *
   * <p>For directed graphs, this is equal to {@code inDegree(node) + outDegree(node)}.
   *
   * <p>For undirected graphs, this is equal to {@code incidentEdges(node).size()} + (number of
   * self-loops incident to {@code node}).
   *
