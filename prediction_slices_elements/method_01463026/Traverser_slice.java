// Source-based slice around line 240
// Method: <com.google.common.graph.Traverser: Iterable breadthFirst(Iterable)>

   * Returns an unmodifiable {@code Iterable} over the nodes reachable from any of the {@code
   * startNodes}, in the order of a breadth-first traversal. This is equivalent to a breadth-first
   * traversal of a graph with an additional root node whose successors are the listed {@code
   * startNodes}.
   *
   * @throws IllegalArgumentException if any of {@code startNodes} is not an element of the graph
   * @see #breadthFirst(Object)
   * @since 24.1
   */
  public final Iterable<N> breadthFirst(Iterable<? extends N> startNodes) {
    ImmutableSet<N> validated = validate(startNodes);
    return () -> newTraversal().breadthFirst(validated.iterator());
  }

  /**
   * Returns an unmodifiable {@code Iterable} over the nodes reachable from {@code startNode}, in
   * the order of a depth-first pre-order traversal. "Pre-order" implies that nodes appear in the
   * {@code Iterable} in the order in which they are first visited.
   *
   * <p><b>Example:</b> The following graph with {@code startNode} {@code a} would return nodes in
