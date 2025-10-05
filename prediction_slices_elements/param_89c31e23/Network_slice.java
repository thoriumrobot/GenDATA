// Source-based slice around line 272
// Method: <com.google.common.graph.Network: Set inEdges(N)>

   *   <li>{@code view.equals(view)} evaluates to {@code true} (but any other {@code equals()}
   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  Set<E> inEdges(N node);

  /**
   * Returns a live view of all edges in this network which can be traversed in the direction (if
   * any) of the edge starting from {@code node}.
   *
   * <p>In a directed network, an outgoing edge's {@link EndpointPair#source()} equals {@code node}.
   *
   * <p>In an undirected network, this is equivalent to {@link #incidentEdges(Object)}.
   *
   * <p>If {@code node} is removed from the network after this method is called, the {@code Set}
