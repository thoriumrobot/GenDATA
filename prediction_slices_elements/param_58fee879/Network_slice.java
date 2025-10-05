// Source-based slice around line 387
// Method: <com.google.common.graph.Network: Set edgesConnecting(N,N)>

   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code nodeU} or {@code nodeV} are re-added to the network after having been removed,
   *       {@code view}'s behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
   *     network
   */
  Set<E> edgesConnecting(N nodeU, N nodeV);

  /**
   * Returns a live view of the set of edges that each directly connect {@code endpoints} (in the
   * order, if any, specified by {@code endpoints}).
   *
   * <p>The resulting set of edges will be parallel (i.e. have equal {@link
   * #incidentNodes(Object)}). If this network does not {@link #allowsParallelEdges() allow parallel
   * edges}, the resulting set will contain at most one edge (equivalent to {@code
   * edgeConnecting(endpoints).asSet()}).
   *
