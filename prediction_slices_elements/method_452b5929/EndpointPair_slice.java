// Source-based slice around line 63
// Method: <com.google.common.graph.EndpointPair: EndpointPair of(Graph,N,N)>

  }

  /** Returns an {@link EndpointPair} representing the endpoints of an undirected edge. */
  public static <N> EndpointPair<N> unordered(N nodeU, N nodeV) {
    // Swap nodes on purpose to prevent callers from relying on the "ordering" of an unordered pair.
    return new Unordered<>(nodeV, nodeU);
  }

  /** Returns an {@link EndpointPair} representing the endpoints of an edge in {@code graph}. */
  static <N> EndpointPair<N> of(Graph<?> graph, N nodeU, N nodeV) {
    return graph.isDirected() ? ordered(nodeU, nodeV) : unordered(nodeU, nodeV);
  }

  /** Returns an {@link EndpointPair} representing the endpoints of an edge in {@code network}. */
  static <N> EndpointPair<N> of(Network<?, ?> network, N nodeU, N nodeV) {
    return network.isDirected() ? ordered(nodeU, nodeV) : unordered(nodeU, nodeV);
  }

  /**
   * If this {@link EndpointPair} {@link #isOrdered()}, returns the node which is the source.
