// Source-based slice around line 136
// Method: <com.google.common.graph.EndpointPair: boolean equals(Object)>

    return Iterators.forArray(nodeU, nodeV);
  }

  /**
   * Two ordered {@link EndpointPair}s are equal if their {@link #source()} and {@link #target()}
   * are equal. Two unordered {@link EndpointPair}s are equal if they contain the same nodes. An
   * ordered {@link EndpointPair} is never equal to an unordered {@link EndpointPair}.
   */
  @Override
  public abstract boolean equals(@Nullable Object obj);

  /**
   * The hashcode of an ordered {@link EndpointPair} is equal to {@code Objects.hash(source(),
   * target())}. The hashcode of an unordered {@link EndpointPair} is equal to {@code
   * nodeU().hashCode() + nodeV().hashCode()}.
   */
  @Override
  public abstract int hashCode();

  private static final class Ordered<N> extends EndpointPair<N> {
