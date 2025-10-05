// Source-based slice around line 194
// Method: <com.google.common.graph.NetworkBuilder: MutableNetwork build()>

   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <E1 extends E> NetworkBuilder<N, E1> edgeOrder(ElementOrder<E1> edgeOrder) {
    NetworkBuilder<N, E1> newBuilder = cast();
    newBuilder.edgeOrder = checkNotNull(edgeOrder);
    return newBuilder;
  }

  /** Returns an empty {@link MutableNetwork} with the properties of this {@link NetworkBuilder}. */
  public <N1 extends N, E1 extends E> MutableNetwork<N1, E1> build() {
    return new StandardMutableNetwork<>(this);
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N, E1 extends E> NetworkBuilder<N1, E1> cast() {
    return (NetworkBuilder<N1, E1>) this;
  }
}
