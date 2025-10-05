// Source-based slice around line 296
// Method: <com.google.common.graph.Network: Set outEdges(N)>

   *   <li>{@code view.equals(view)} evaluates to {@code true} (but any other {@code equals()}
   *       expression involving {@code view} will throw)
   *   <li>{@code hashCode()} does not throw
   *   <li>if {@code node} is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  Set<E> outEdges(N node);

  /**
   * Returns the count of {@code node}'s {@link #incidentEdges(Object) incident edges}, counting
   * self-loops twice (equivalently, the number of times an edge touches {@code node}).
   *
   * <p>For directed networks, this is equal to {@code inDegree(node) + outDegree(node)}.
   *
   * <p>For undirected networks, this is equal to {@code incidentEdges(node).size()} + (number of
   * self-loops incident to {@code node}).
   *
