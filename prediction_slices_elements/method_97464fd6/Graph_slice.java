// Source-based slice around line 129
// Method: <com.google.common.graph.Graph: boolean isDirected()>

  // Graph properties
  //

  /**
   * Returns true if the edges in this graph are directed. Directed edges connect a {@link
   * EndpointPair#source() source node} to a {@link EndpointPair#target() target node}, while
   * undirected edges connect a pair of nodes to each other.
   */
  @Override
  boolean isDirected();

  /**
   * Returns true if this graph allows self-loops (edges that connect a node to itself). Attempting
   * to add a self-loop to a graph that does not allow them will throw an {@link
   * IllegalArgumentException}.
   */
  @Override
  boolean allowsSelfLoops();

  /** Returns the order of iteration for the elements of {@link #nodes()}. */
