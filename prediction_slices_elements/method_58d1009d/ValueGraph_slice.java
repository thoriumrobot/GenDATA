// Source-based slice around line 128
// Method: <com.google.common.graph.ValueGraph: Graph asGraph()>


  /** Returns all edges in this graph. */
  @Override
  Set<EndpointPair<N>> edges();

  /**
   * Returns a live view of this graph as a {@link Graph}. The resulting {@link Graph} will have an
   * edge connecting node A to node B if this {@link ValueGraph} has an edge connecting A to B.
   */
  Graph<N> asGraph();

  //
  // ValueGraph properties
  //

  /**
   * Returns true if the edges in this graph are directed. Directed edges connect a {@link
   * EndpointPair#source() source node} to a {@link EndpointPair#target() target node}, while
   * undirected edges connect a pair of nodes to each other.
   */
