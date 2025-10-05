// Source-based slice around line 525
// Method: <com.google.common.graph.Graphs: MutableGraph inducedSubgraph(Graph,Iterable)>

  // Graph copy methods

  /**
   * Returns the subgraph of {@code graph} induced by {@code nodes}. This subgraph is a new graph
   * that contains all of the nodes in {@code nodes}, and all of the {@link Graph#edges() edges}
   * from {@code graph} for which both nodes are contained by {@code nodes}.
   *
   * @throws IllegalArgumentException if any element in {@code nodes} is not a node in the graph
   */
  public static <N> MutableGraph<N> inducedSubgraph(Graph<N> graph, Iterable<? extends N> nodes) {
    MutableGraph<N> subgraph =
        (nodes instanceof Collection)
            ? GraphBuilder.from(graph).expectedNodeCount(((Collection) nodes).size()).build()
            : GraphBuilder.from(graph).build();
    for (N node : nodes) {
      subgraph.addNode(node);
    }
    for (N node : subgraph.nodes()) {
      for (N successorNode : graph.successors(node)) {
        if (subgraph.nodes().contains(successorNode)) {
