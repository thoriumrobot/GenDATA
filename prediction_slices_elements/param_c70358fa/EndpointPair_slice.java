// Source-based slice around line 52
// Method: <com.google.common.graph.EndpointPair: EndpointPair ordered(N,N)>

  private final N nodeU;
  private final N nodeV;

  private EndpointPair(N nodeU, N nodeV) {
    this.nodeU = checkNotNull(nodeU);
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
