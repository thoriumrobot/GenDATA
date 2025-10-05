// Source-based slice around line 139
// Method: <com.google.common.graph.Network: boolean isDirected()>

  //
  // Network properties
  //

  /**
   * Returns true if the edges in this network are directed. Directed edges connect a {@link
   * EndpointPair#source() source node} to a {@link EndpointPair#target() target node}, while
   * undirected edges connect a pair of nodes to each other.
   */
  boolean isDirected();

  /**
   * Returns true if this network allows parallel edges. Attempting to add a parallel edge to a
   * network that does not allow them will throw an {@link IllegalArgumentException}.
   */
  boolean allowsParallelEdges();

  /**
   * Returns true if this network allows self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a network that does not allow them will throw an {@link
