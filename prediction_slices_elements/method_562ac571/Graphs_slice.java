// Source-based slice around line 551
// Method: <com.google.common.graph.Graphs: MutableValueGraph inducedSubgraph(ValueGraph,Iterable)>


  /**
   * Returns the subgraph of {@code graph} induced by {@code nodes}. This subgraph is a new graph
   * that contains all of the nodes in {@code nodes}, and all of the {@link Graph#edges() edges}
   * (and associated edge values) from {@code graph} for which both nodes are contained by {@code
   * nodes}.
   *
   * @throws IllegalArgumentException if any element in {@code nodes} is not a node in the graph
   */
  public static <N, V> MutableValueGraph<N, V> inducedSubgraph(
      ValueGraph<N, V> graph, Iterable<? extends N> nodes) {
    MutableValueGraph<N, V> subgraph =
        (nodes instanceof Collection)
            ? ValueGraphBuilder.from(graph).expectedNodeCount(((Collection) nodes).size()).build()
            : ValueGraphBuilder.from(graph).build();
    for (N node : nodes) {
      subgraph.addNode(node);
    }
    for (N node : subgraph.nodes()) {
      for (N successorNode : graph.successors(node)) {
