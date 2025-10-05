// Source-based slice around line 225
// Method: <com.google.common.graph.Network: Set successors(N)>

   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  @Override
  Set<N> successors(N node);

  /**
   * Returns a live view of the edges whose {@link #incidentNodes(Object) incident nodes} in this
   * network include {@code node}.
   *
   * <p>This is equal to the union of {@link #inEdges(Object)} and {@link #outEdges(Object)}.
   *
   * <p>If {@code node} is removed from the network after this method is called, the {@code Set}
   * {@code view} returned by this method will be invalidated, and will throw {@code
   * IllegalStateException} if it is accessed in any way, with the following exceptions:
