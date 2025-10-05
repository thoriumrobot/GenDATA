// Source-based slice around line 70
// Method: <com.google.common.graph.ImmutableValueGraph: ImmutableValueGraph copyOf(ImmutableValueGraph)>

  /**
   * Simply returns its argument.
   *
   * @deprecated no need to use this
   */
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
