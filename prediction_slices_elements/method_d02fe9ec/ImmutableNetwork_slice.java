// Source-based slice around line 73
// Method: <com.google.common.graph.ImmutableNetwork: ImmutableNetwork copyOf(ImmutableNetwork)>

  /**
   * Simply returns its argument.
   *
   * @deprecated no need to use this
   */
  @InlineMe(
      replacement = "checkNotNull(network)",
      staticImports = "com.google.common.base.Preconditions.checkNotNull")
  @Deprecated
  public static <N, E> ImmutableNetwork<N, E> copyOf(ImmutableNetwork<N, E> network) {
    return checkNotNull(network);
  }

  @Override
  public ImmutableGraph<N> asGraph() {
    return new ImmutableGraph<>(super.asGraph()); // safe because the view is effectively immutable
  }

  private static <N, E> Map<N, NetworkConnections<N, E>> getNodeConnections(Network<N, E> network) {
    // ImmutableMap.Builder maintains the order of the elements as inserted, so the map will have
