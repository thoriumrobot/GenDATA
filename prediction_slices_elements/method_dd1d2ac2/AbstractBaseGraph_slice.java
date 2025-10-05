// Source-based slice around line 67
// Method: <com.google.common.graph.AbstractBaseGraph: Set edges()>

    checkState((degreeSum & 1) == 0);
    return degreeSum >>> 1;
  }

  /**
   * An implementation of {@link BaseGraph#edges()} defined in terms of {@link Graph#nodes()} and
   * {@link #successors(Object)}.
   */
  @Override
  public Set<EndpointPair<N>> edges() {
    return new AbstractSet<EndpointPair<N>>() {
      @Override
      public UnmodifiableIterator<EndpointPair<N>> iterator() {
        return EndpointPairIterator.of(AbstractBaseGraph.this);
      }

      @Override
      public int size() {
        return Ints.saturatedCast(edgeCount());
      }
