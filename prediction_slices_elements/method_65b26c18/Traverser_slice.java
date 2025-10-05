// Source-based slice around line 226
// Method: <com.google.common.graph.Traverser: Iterable breadthFirst(N)>

   * {@snippet :
   * Iterables.limit(Traverser.forGraph(graph).breadthFirst(node), maxNumberOfNodes);
   * }
   *
   * <p>See <a href="https://en.wikipedia.org/wiki/Breadth-first_search">Wikipedia</a> for more
   * info.
   *
   * @throws IllegalArgumentException if {@code startNode} is not an element of the graph
   */
  public final Iterable<N> breadthFirst(N startNode) {
    return breadthFirst(ImmutableSet.of(startNode));
  }

  /**
   * Returns an unmodifiable {@code Iterable} over the nodes reachable from any of the {@code
   * startNodes}, in the order of a breadth-first traversal. This is equivalent to a breadth-first
   * traversal of a graph with an additional root node whose successors are the listed {@code
   * startNodes}.
   *
   * @throws IllegalArgumentException if any of {@code startNodes} is not an element of the graph
