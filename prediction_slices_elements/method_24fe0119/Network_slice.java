// Source-based slice around line 248
// Method: <com.google.common.graph.Network: Set incidentEdges(N)>

   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   * @since 24.0
   */
  Set<E> incidentEdges(N node);

  /**
   * Returns a live view of all edges in this network which can be traversed in the direction (if
   * any) of the edge to end at {@code node}.
   *
   * <p>In a directed network, an incoming edge's {@link EndpointPair#target()} equals {@code node}.
   *
   * <p>In an undirected network, this is equivalent to {@link #incidentEdges(Object)}.
   *
   * <p>If {@code node} is removed from the network after this method is called, the {@code Set}
