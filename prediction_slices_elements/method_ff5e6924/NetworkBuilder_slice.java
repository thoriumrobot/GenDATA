// Source-based slice around line 104
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder from(Network)>


  /**
   * Returns a {@link NetworkBuilder} initialized with all properties queryable from {@code
   * network}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link Network} interface,
   * such as {@link Network#isDirected()}. Other properties, such as {@link
   * #expectedNodeCount(int)}, are not set in the new builder.
   */
  public static <N, E> NetworkBuilder<N, E> from(Network<N, E> network) {
    return new NetworkBuilder<N, E>(network.isDirected())
        .allowsParallelEdges(network.allowsParallelEdges())
        .allowsSelfLoops(network.allowsSelfLoops())
        .nodeOrder(network.nodeOrder())
        .edgeOrder(network.edgeOrder());
  }

  /**
   * Returns an {@link ImmutableNetwork.Builder} with the properties of this {@link NetworkBuilder}.
   *
