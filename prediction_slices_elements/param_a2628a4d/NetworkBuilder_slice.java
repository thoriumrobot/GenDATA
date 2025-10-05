// Source-based slice around line 131
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder allowsParallelEdges(boolean)>

  }

  /**
   * Specifies whether the network will allow parallel edges. Attempting to add a parallel edge to a
   * network that does not allow them will throw an {@link UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
   */
  @CanIgnoreReturnValue
  public NetworkBuilder<N, E> allowsParallelEdges(boolean allowsParallelEdges) {
    this.allowsParallelEdges = allowsParallelEdges;
    return this;
  }

  /**
   * Specifies whether the network will allow self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a network that does not allow them will throw an {@link
   * UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
