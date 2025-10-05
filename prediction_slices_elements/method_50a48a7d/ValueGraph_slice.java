// Source-based slice around line 230
// Method: <com.google.common.graph.ValueGraph: Set successors(N)>

   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the graph after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  @Override
  Set<N> successors(N node);

  /**
   * Returns a live view of the edges in this graph whose endpoints include {@code node}.
   *
   * <p>This is equal to the union of incoming and outgoing edges.
   *
   * <p>If {@code node} is removed from the graph after this method is called, the {@code Set}
   * {@code view} returned by this method will be invalidated, and will throw {@code
   * IllegalStateException} if it is accessed in any way, with the following exceptions:
   *
