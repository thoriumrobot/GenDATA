// Source-based slice around line 340
// Method: <com.google.common.graph.Traverser: Iterable depthFirstPostOrder(Iterable)>

   * Returns an unmodifiable {@code Iterable} over the nodes reachable from any of the {@code
   * startNodes}, in the order of a depth-first post-order traversal. This is equivalent to a
   * depth-first post-order traversal of a graph with an additional root node whose successors are
   * the listed {@code startNodes}.
   *
   * @throws IllegalArgumentException if any of {@code startNodes} is not an element of the graph
   * @see #depthFirstPostOrder(Object)
   * @since 24.1
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
