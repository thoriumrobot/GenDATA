// Source-based slice around line 298
// Method: <com.google.common.graph.Graphs: EndpointPair transpose(EndpointPair)>

    }

    if (network instanceof TransposedNetwork) {
      return ((TransposedNetwork<N, E>) network).network;
    }

    return new TransposedNetwork<>(network);
  }

  static <N> EndpointPair<N> transpose(EndpointPair<N> endpoints) {
    if (endpoints.isOrdered()) {
      return EndpointPair.ordered(endpoints.target(), endpoints.source());
    }
    return endpoints;
  }

  // NOTE: this should work as long as the delegate graph's implementation of edges() (like that of
  // AbstractGraph) derives its behavior from calling successors().
  private static final class TransposedGraph<N> extends ForwardingGraph<N> {
    private final Graph<N> graph;
