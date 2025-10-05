// Source-based slice around line 286
// Method: <com.google.common.graph.Graphs: Network transpose(Network)>

    }

    return new TransposedValueGraph<>(graph);
  }

  /**
   * Returns a view of {@code network} with the direction (if any) of every edge reversed. All other
   * properties remain intact, and further updates to {@code network} will be reflected in the view.
   */
  public static <N, E> Network<N, E> transpose(Network<N, E> network) {
    if (!network.isDirected()) {
      return network; // the transpose of an undirected network is an identical network
    }

    if (network instanceof TransposedNetwork) {
      return ((TransposedNetwork<N, E>) network).network;
    }

    return new TransposedNetwork<>(network);
  }
