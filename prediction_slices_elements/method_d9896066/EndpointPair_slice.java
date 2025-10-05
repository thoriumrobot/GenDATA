// Source-based slice around line 144
// Method: <com.google.common.graph.EndpointPair: int hashCode()>

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
    private Ordered(N source, N target) {
      super(source, target);
    }

    @Override
    public N source() {
      return nodeU();
    }
