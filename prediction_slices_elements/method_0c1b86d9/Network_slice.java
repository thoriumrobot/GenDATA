// Source-based slice around line 358
// Method: <com.google.common.graph.Network: Set adjacentEdges(E)>

   *   <li>{@code view.equals(view)} evaluates to {@code true} (but any other {@code equals()}
   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code edge} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code edge} is not an element of this network
   */
  Set<E> adjacentEdges(E edge);

  /**
   * Returns a live view of the set of edges that each directly connect {@code nodeU} to {@code
   * nodeV}.
   *
   * <p>In an undirected network, this is equal to {@code edgesConnecting(nodeV, nodeU)}.
   *
   * <p>The resulting set of edges will be parallel (i.e. have equal {@link
   * #incidentNodes(Object)}). If this network does not {@link #allowsParallelEdges() allow parallel
   * edges}, the resulting set will contain at most one edge (equivalent to {@code
