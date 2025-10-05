// Source-based slice around line 75
// Method: <com.google.common.graph.ImmutableGraph: ImmutableGraph copyOf(ImmutableGraph)>

  /**
   * Simply returns its argument.
   *
   * @deprecated no need to use this
   */
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
