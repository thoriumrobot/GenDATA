// Source-based slice around line 139
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder expectedNodeCount(int)>

    return this;
  }

  /**
   * Specifies the expected number of nodes in the graph.
   *
   * @throws IllegalArgumentException if {@code expectedNodeCount} is negative
   */
  @CanIgnoreReturnValue
  public GraphBuilder<N> expectedNodeCount(int expectedNodeCount) {
    this.expectedNodeCount = Optional.of(checkNonNegative(expectedNodeCount));
    return this;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Graph#nodes()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <N1 extends N> GraphBuilder<N1> nodeOrder(ElementOrder<N1> nodeOrder) {
