// Source-based slice around line 80
// Method: <com.google.common.graph.ImmutableGraph: ElementOrder incidentEdgeOrder()>

  @InlineMe(
      replacement = "checkNotNull(graph)",
      staticImports = "com.google.common.base.Preconditions.checkNotNull")
  @Deprecated
  public static <N> ImmutableGraph<N> copyOf(ImmutableGraph<N> graph) {
    return checkNotNull(graph);
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return ElementOrder.stable();
  }

  private static <N> ImmutableMap<N, GraphConnections<N, Presence>> getNodeConnections(
      Graph<N> graph) {
    // ImmutableMap.Builder maintains the order of the elements as inserted, so the map will have
    // whatever ordering the graph's nodes do, so ImmutableSortedMap is unnecessary even if the
    // input nodes are sorted.
    ImmutableMap.Builder<N, GraphConnections<N, Presence>> nodeConnections = ImmutableMap.builder();
    for (N node : graph.nodes()) {
