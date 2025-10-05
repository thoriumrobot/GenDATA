// Source-based slice around line 44
// Method: <com.google.common.graph.ForwardingValueGraph: long edgeCount()>

  public Set<N> nodes() {
    return delegate().nodes();
  }

  /**
   * Defer to {@link AbstractValueGraph#edges()} (based on {@link #successors(Object)}) for full
   * edges() implementation.
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
