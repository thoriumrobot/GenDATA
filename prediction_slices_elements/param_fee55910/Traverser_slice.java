// Source-based slice around line 180
// Method: <com.google.common.graph.Traverser: Traverser forTree(SuccessorsFunction)>

   * leftChild(node)} and {@code rightChild(node)}, this method can be called as
   *
   * {@snippet :
   * Traverser.forTree(node -> ImmutableList.of(leftChild(node), rightChild(node)));
   * }
   *
   * @param tree {@link SuccessorsFunction} representing a directed acyclic graph that has at most
   *     one path between any two nodes
   */
  public static <N> Traverser<N> forTree(SuccessorsFunction<N> tree) {
    if (tree instanceof BaseGraph) {
      checkArgument(((BaseGraph<?>) tree).isDirected(), "Undirected graphs can never be trees.");
    }
    if (tree instanceof Network) {
      checkArgument(((Network<?, ?>) tree).isDirected(), "Undirected networks can never be trees.");
    }
    return new Traverser<N>(tree) {
      @Override
      Traversal<N> newTraversal() {
        return Traversal.inTree(tree);
