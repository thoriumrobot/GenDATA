// Source-based slice around line 66
// Method: <com.google.common.graph.BaseGraph: ElementOrder incidentEdgeOrder()>

  ElementOrder<N> nodeOrder();

  /**
   * Returns an {@link ElementOrder} that specifies the order of iteration for the elements of
   * {@link #edges()}, {@link #adjacentNodes(Object)}, {@link #predecessors(Object)}, {@link
   * #successors(Object)} and {@link #incidentEdges(Object)}.
   *
   * @since 29.0
   */
  ElementOrder<N> incidentEdgeOrder();

  //
  // Element-level accessors
  //

  /**
   * Returns a live view of the nodes which have an incident edge in common with {@code node} in
   * this graph.
   *
   * <p>This is equal to the union of {@link #predecessors(Object)} and {@link #successors(Object)}.
