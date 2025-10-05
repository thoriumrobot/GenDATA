// Source-based slice around line 45
// Method: <com.google.common.graph.ForwardingNetwork: boolean isDirected()>

    return delegate().nodes();
  }

  @Override
  public Set<E> edges() {
    return delegate().edges();
  }

  @Override
  public boolean isDirected() {
    return delegate().isDirected();
  }

  @Override
  public boolean allowsParallelEdges() {
    return delegate().allowsParallelEdges();
  }

  @Override
  public boolean allowsSelfLoops() {
