// Source-based slice around line 117
// Method: <com.google.common.graph.Graph: Set edges()>

  // Graph-level accessors
  //

  /** Returns all nodes in this graph, in the order specified by {@link #nodeOrder()}. */
  @Override
  Set<N> nodes();

  /** Returns all edges in this graph. */
  @Override
  Set<EndpointPair<N>> edges();

  //
  // Graph properties
  //

  /**
   * Returns true if the edges in this graph are directed. Directed edges connect a {@link
   * EndpointPair#source() source node} to a {@link EndpointPair#target() target node}, while
   * undirected edges connect a pair of nodes to each other.
   */
