// Source-based slice around line 348
// Method: <com.google.common.graph.Traverser: ImmutableSet validate(Iterable)>

   */
  public final Iterable<N> depthFirstPostOrder(Iterable<? extends N> startNodes) {
    ImmutableSet<N> validated = validate(startNodes);
    return () -> newTraversal().postOrder(validated.iterator());
  }

  abstract Traversal<N> newTraversal();

  @SuppressWarnings("CheckReturnValue")
  private ImmutableSet<N> validate(Iterable<? extends N> startNodes) {
    ImmutableSet<N> copy = ImmutableSet.copyOf(startNodes);
    for (N node : copy) {
      successorFunction.successors(node); // Will throw if node doesn't exist
    }
    return copy;
  }

  /**
   * Abstracts away the difference between traversing a graph vs. a tree. For a tree, we just take
   * the next element from the next non-empty iterator; for graph, we need to loop through the next
