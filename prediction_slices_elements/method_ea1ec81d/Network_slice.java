// Source-based slice around line 128
// Method: <com.google.common.graph.Network: Graph asGraph()>


  /**
   * Returns a live view of this network as a {@link Graph}. The resulting {@link Graph} will have
   * an edge connecting node A to node B if this {@link Network} has an edge connecting A to B.
   *
   * <p>If this network {@link #allowsParallelEdges() allows parallel edges}, parallel edges will be
   * treated as if collapsed into a single edge. For example, the {@link #degree(Object)} of a node
   * in the {@link Graph} view may be less than the degree of the same node in this {@link Network}.
   */
  Graph<N> asGraph();

  //
  // Network properties
  //

  /**
   * Returns true if the edges in this network are directed. Directed edges connect a {@link
   * EndpointPair#source() source node} to a {@link EndpointPair#target() target node}, while
   * undirected edges connect a pair of nodes to each other.
   */
