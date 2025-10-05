// Source-based slice around line 241
// Method: <com.google.common.graph.Graphs: ImmutableSet reachableNodes(Graph,N)>

   * section of the Guava User's Guide</a> for more information.
   *
   * <p>The {@link Set} returned is a "snapshot" based on the current topology of {@code graph},
   * rather than a live view. In other words, modifications to {@code graph} made after this method
   * returns will not be reflected in the set.
   *
   * @throws IllegalArgumentException if {@code node} is not present in {@code graph}
   * @since 33.1.0 (present with return type {@code Set} since 20.0)
   */
  public static <N> ImmutableSet<N> reachableNodes(Graph<N> graph, N node) {
    checkArgument(graph.nodes().contains(node), NODE_NOT_IN_GRAPH, node);
    return ImmutableSet.copyOf(Traverser.forGraph(graph).breadthFirst(node));
  }

  // Graph mutation methods

  // Graph view methods

  /**
   * Returns a view of {@code graph} with the direction (if any) of every edge reversed. All other
