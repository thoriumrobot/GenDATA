// Source-based slice around line 137
// Method: <com.google.common.graph.Graph: boolean allowsSelfLoops()>

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
  @Override
  ElementOrder<N> nodeOrder();

  /**
   * Returns an {@link ElementOrder} that specifies the order of iteration for the elements of
   * {@link #edges()}, {@link #adjacentNodes(Object)}, {@link #predecessors(Object)}, {@link
   * #successors(Object)} and {@link #incidentEdges(Object)}.
   *
