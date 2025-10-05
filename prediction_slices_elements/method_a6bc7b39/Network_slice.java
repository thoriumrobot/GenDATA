// Source-based slice around line 152
// Method: <com.google.common.graph.Network: boolean allowsSelfLoops()>

   * network that does not allow them will throw an {@link IllegalArgumentException}.
   */
  boolean allowsParallelEdges();

  /**
   * Returns true if this network allows self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a network that does not allow them will throw an {@link
   * IllegalArgumentException}.
   */
  boolean allowsSelfLoops();

  /** Returns the order of iteration for the elements of {@link #nodes()}. */
  ElementOrder<N> nodeOrder();

  /** Returns the order of iteration for the elements of {@link #edges()}. */
  ElementOrder<E> edgeOrder();

  //
  // Element-level accessors
  //
