// Source-based slice around line 290
// Method: <com.google.common.graph.Traverser: Iterable depthFirstPreOrder(Iterable)>

   * Returns an unmodifiable {@code Iterable} over the nodes reachable from any of the {@code
   * startNodes}, in the order of a depth-first pre-order traversal. This is equivalent to a
   * depth-first pre-order traversal of a graph with an additional root node whose successors are
   * the listed {@code startNodes}.
   *
   * @throws IllegalArgumentException if any of {@code startNodes} is not an element of the graph
   * @see #depthFirstPreOrder(Object)
   * @since 24.1
   */
  public final Iterable<N> depthFirstPreOrder(Iterable<? extends N> startNodes) {
    ImmutableSet<N> validated = validate(startNodes);
    return () -> newTraversal().preOrder(validated.iterator());
  }

  /**
   * Returns an unmodifiable {@code Iterable} over the nodes reachable from {@code startNode}, in
   * the order of a depth-first post-order traversal. "Post-order" implies that nodes appear in the
   * {@code Iterable} in the order in which they are visited for the last time.
   *
   * <p><b>Example:</b> The following graph with {@code startNode} {@code a} would return nodes in
