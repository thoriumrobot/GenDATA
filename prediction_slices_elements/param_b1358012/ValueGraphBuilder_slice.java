// Source-based slice around line 132
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder allowsSelfLoops(boolean)>


  /**
   * Specifies whether the graph will allow self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a graph that does not allow them will throw an {@link
   * UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
   */
  @CanIgnoreReturnValue
  public ValueGraphBuilder<N, V> allowsSelfLoops(boolean allowsSelfLoops) {
    this.allowsSelfLoops = allowsSelfLoops;
    return this;
  }

  /**
   * Specifies the expected number of nodes in the graph.
   *
   * @throws IllegalArgumentException if {@code expectedNodeCount} is negative
   */
  @CanIgnoreReturnValue
