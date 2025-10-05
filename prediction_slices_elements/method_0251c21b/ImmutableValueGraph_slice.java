// Source-based slice around line 75
// Method: <com.google.common.graph.ImmutableValueGraph: ElementOrder incidentEdgeOrder()>

  @InlineMe(
      replacement = "checkNotNull(graph)",
      staticImports = "com.google.common.base.Preconditions.checkNotNull")
  @Deprecated
  public static <N, V> ImmutableValueGraph<N, V> copyOf(ImmutableValueGraph<N, V> graph) {
    return checkNotNull(graph);
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return ElementOrder.stable();
  }

  @Override
  public ImmutableGraph<N> asGraph() {
    return new ImmutableGraph<>(this); // safe because the view is effectively immutable
  }

  private static <N, V> ImmutableMap<N, GraphConnections<N, V>> getNodeConnections(
      ValueGraph<N, V> graph) {
