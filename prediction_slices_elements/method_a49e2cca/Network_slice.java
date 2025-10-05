// Source-based slice around line 158
// Method: <com.google.common.graph.Network: ElementOrder edgeOrder()>

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

  /**
   * Returns a live view of the nodes which have an incident edge in common with {@code node} in
   * this graph.
   *
   * <p>This is equal to the union of {@link #predecessors(Object)} and {@link #successors(Object)}.
