// Source-based slice around line 144
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder allowsSelfLoops(boolean)>


  /**
   * Specifies whether the network will allow self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a network that does not allow them will throw an {@link
   * UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
   */
  @CanIgnoreReturnValue
  public NetworkBuilder<N, E> allowsSelfLoops(boolean allowsSelfLoops) {
    this.allowsSelfLoops = allowsSelfLoops;
    return this;
  }

  /**
   * Specifies the expected number of nodes in the network.
   *
   * @throws IllegalArgumentException if {@code expectedNodeCount} is negative
   */
  @CanIgnoreReturnValue
