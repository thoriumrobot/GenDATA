// Source-based slice around line 166
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder expectedEdgeCount(int)>

    return this;
  }

  /**
   * Specifies the expected number of edges in the network.
   *
   * @throws IllegalArgumentException if {@code expectedEdgeCount} is negative
   */
  @CanIgnoreReturnValue
  public NetworkBuilder<N, E> expectedEdgeCount(int expectedEdgeCount) {
    this.expectedEdgeCount = Optional.of(checkNonNegative(expectedEdgeCount));
    return this;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Network#nodes()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <N1 extends N> NetworkBuilder<N1, E> nodeOrder(ElementOrder<N1> nodeOrder) {
