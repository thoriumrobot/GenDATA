// Source-based slice around line 41
// Method: <com.google.common.graph.ForwardingGraph: long edgeCount()>

  public Set<N> nodes() {
    return delegate().nodes();
  }

  /**
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
