// Source-based slice around line 331
// Method: <com.google.common.graph.Network: int outDegree(N)>


  /**
   * Returns the count of {@code node}'s {@link #outEdges(Object) outgoing edges} in a directed
   * network. In an undirected network, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
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
