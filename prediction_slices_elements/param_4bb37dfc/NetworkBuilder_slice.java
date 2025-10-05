// Source-based slice around line 155
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder expectedNodeCount(int)>

    return this;
  }

  /**
   * Specifies the expected number of nodes in the network.
   *
   * @throws IllegalArgumentException if {@code expectedNodeCount} is negative
   */
  @CanIgnoreReturnValue
  public NetworkBuilder<N, E> expectedNodeCount(int expectedNodeCount) {
    this.expectedNodeCount = Optional.of(checkNonNegative(expectedNodeCount));
    return this;
  }

  /**
   * Specifies the expected number of edges in the network.
   *
   * @throws IllegalArgumentException if {@code expectedEdgeCount} is negative
   */
  @CanIgnoreReturnValue
