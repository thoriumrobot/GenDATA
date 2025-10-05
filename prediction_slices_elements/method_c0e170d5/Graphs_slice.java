// Source-based slice around line 189
// Method: <com.google.common.graph.Graphs: ImmutableGraph transitiveClosure(Graph)>

   * Object) reachable} from node A.
   *
   * <p>This is a "snapshot" based on the current topology of {@code graph}, rather than a live view
   * of the transitive closure of {@code graph}. In other words, the returned {@link Graph} will not
   * be updated after modifications to {@code graph}.
   *
   * @since 33.1.0 (present with return type {@code Graph} since 20.0)
   */
  // TODO(b/31438252): Consider potential optimizations for this algorithm.
  public static <N> ImmutableGraph<N> transitiveClosure(Graph<N> graph) {
    ImmutableGraph.Builder<N> transitiveClosure =
        GraphBuilder.from(graph).allowsSelfLoops(true).<N>immutable();
    // Every node is, at a minimum, reachable from itself. Since the resulting transitive closure
    // will have no isolated nodes, we can skip adding nodes explicitly and let putEdge() do it.

    if (graph.isDirected()) {
      // Note: works for both directed and undirected graphs, but we only use in the directed case.
      for (N node : graph.nodes()) {
        for (N reachableNode : reachableNodes(graph, node)) {
          transitiveClosure.putEdge(node, reachableNode);
