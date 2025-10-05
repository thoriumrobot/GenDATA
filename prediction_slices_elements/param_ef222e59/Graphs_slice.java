// Source-based slice around line 583
// Method: <com.google.common.graph.Graphs: MutableNetwork inducedSubgraph(Network,Iterable)>

  /**
   * Returns the subgraph of {@code network} induced by {@code nodes}. This subgraph is a new graph
   * that contains all of the nodes in {@code nodes}, and all of the {@link Network#edges() edges}
   * from {@code network} for which the {@link Network#incidentNodes(Object) incident nodes} are
   * both contained by {@code nodes}.
   *
   * @throws IllegalArgumentException if any element in {@code nodes} is not a node in the graph
   */
  public static <N, E> MutableNetwork<N, E> inducedSubgraph(
      Network<N, E> network, Iterable<? extends N> nodes) {
    MutableNetwork<N, E> subgraph =
        (nodes instanceof Collection)
            ? NetworkBuilder.from(network).expectedNodeCount(((Collection) nodes).size()).build()
            : NetworkBuilder.from(network).build();
    for (N node : nodes) {
      subgraph.addNode(node);
    }
    for (N node : subgraph.nodes()) {
      for (E edge : network.outEdges(node)) {
        N successorNode = network.incidentNodes(edge).adjacentNode(node);
