// Source-based slice around line 338
// Method: <com.google.common.graph.Network: EndpointPair incidentNodes(E)>

   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  int outDegree(N node);

  /**
   * Returns the nodes which are the endpoints of {@code edge} in this network.
   *
   * @throws IllegalArgumentException if {@code edge} is not an element of this network
   */
  EndpointPair<N> incidentNodes(E edge);

  /**
   * Returns a live view of the edges which have an {@link #incidentNodes(Object) incident node} in
   * common with {@code edge}. An edge is not considered adjacent to itself.
   *
   * <p>If {@code edge} is removed from the network after this method is called, the {@code Set}
   * {@code view} returned by this method will be invalidated, and will throw {@code
   * IllegalStateException} if it is accessed in any way, with the following exceptions:
   *
   * <ul>
