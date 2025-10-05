// Source-based slice around line 46
// Method: <com.google.common.graph.ForwardingGraph: boolean isDirected()>

   * Defer to {@link AbstractGraph#edges()} (based on {@link #successors(Object)}) for full edges()
   * implementation.
   */
  @Override
  protected long edgeCount() {
    return delegate().edges().size();
  }

  @Override
  public boolean isDirected() {
    return delegate().isDirected();
  }

  @Override
  public boolean allowsSelfLoops() {
    return delegate().allowsSelfLoops();
  }

  @Override
  public ElementOrder<N> nodeOrder() {
