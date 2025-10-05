// Source-based slice around line 187
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder edgeOrder(ElementOrder)>

    newBuilder.nodeOrder = checkNotNull(nodeOrder);
    return newBuilder;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Network#edges()}.
   *
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

