// Source-based slice around line 176
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder nodeOrder(ElementOrder)>

    this.expectedEdgeCount = Optional.of(checkNonNegative(expectedEdgeCount));
    return this;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Network#nodes()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <N1 extends N> NetworkBuilder<N1, E> nodeOrder(ElementOrder<N1> nodeOrder) {
    NetworkBuilder<N1, E> newBuilder = cast();
    newBuilder.nodeOrder = checkNotNull(nodeOrder);
    return newBuilder;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Network#edges()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
