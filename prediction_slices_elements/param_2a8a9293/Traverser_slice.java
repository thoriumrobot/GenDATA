// Source-based slice around line 276
// Method: <com.google.common.graph.Traverser: Iterable depthFirstPreOrder(N)>

   * {@snippet :
   * Iterables.limit(
   *     Traverser.forGraph(graph).depthFirstPreOrder(node), maxNumberOfNodes);
   * }
   *
   * <p>See <a href="https://en.wikipedia.org/wiki/Depth-first_search">Wikipedia</a> for more info.
   *
   * @throws IllegalArgumentException if {@code startNode} is not an element of the graph
   */
  public final Iterable<N> depthFirstPreOrder(N startNode) {
    return depthFirstPreOrder(ImmutableSet.of(startNode));
  }

  /**
   * Returns an unmodifiable {@code Iterable} over the nodes reachable from any of the {@code
   * startNodes}, in the order of a depth-first pre-order traversal. This is equivalent to a
   * depth-first pre-order traversal of a graph with an additional root node whose successors are
   * the listed {@code startNodes}.
   *
   * @throws IllegalArgumentException if any of {@code startNodes} is not an element of the graph
