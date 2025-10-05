// Source-based slice around line 126
// Method: <com.google.common.graph.EndpointPair: UnmodifiableIterator iterator()>


  /**
   * Returns {@code true} if this {@link EndpointPair} is an ordered pair (i.e. represents the
   * endpoints of a directed edge).
   */
  public abstract boolean isOrdered();

  /** Iterates in the order {@link #nodeU()}, {@link #nodeV()}. */
  @Override
  public final UnmodifiableIterator<N> iterator() {
    return Iterators.forArray(nodeU, nodeV);
  }

  /**
   * Two ordered {@link EndpointPair}s are equal if their {@link #source()} and {@link #target()}
   * are equal. Two unordered {@link EndpointPair}s are equal if they contain the same nodes. An
   * ordered {@link EndpointPair} is never equal to an unordered {@link EndpointPair}.
   */
  @Override
  public abstract boolean equals(@Nullable Object obj);
