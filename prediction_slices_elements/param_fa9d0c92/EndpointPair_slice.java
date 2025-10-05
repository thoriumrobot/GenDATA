// Source-based slice around line 57
// Method: <com.google.common.graph.EndpointPair: EndpointPair unordered(N,N)>

    this.nodeV = checkNotNull(nodeV);
  }

  /** Returns an {@link EndpointPair} representing the endpoints of a directed edge. */
  public static <N> EndpointPair<N> ordered(N source, N target) {
    return new Ordered<>(source, target);
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
