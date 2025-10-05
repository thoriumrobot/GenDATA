// Source-based slice around line 184
// Method: <com.google.common.graph.Network: Set adjacentNodes(N)>

   *   <li>{@code view.equals(view)} evaluates to {@code true} (but any other {@code equals()}
   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  Set<N> adjacentNodes(N node);

  /**
   * Returns a live view of all nodes in this network adjacent to {@code node} which can be reached
   * by traversing {@code node}'s incoming edges <i>against</i> the direction (if any) of the edge.
   *
   * <p>In an undirected network, this is equivalent to {@link #adjacentNodes(Object)}.
   *
   * <p>If {@code node} is removed from the network after this method is called, the {@code Set}
   * returned by this method will be invalidated, and will throw {@code IllegalStateException} if it
   * is accessed in any way.
