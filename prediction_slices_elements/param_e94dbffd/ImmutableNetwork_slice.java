// Source-based slice around line 121
// Method: <com.google.common.graph.ImmutableNetwork: Function sourceNodeFn(Network)>

    } else {
      Map<E, N> incidentEdgeMap =
          Maps.asMap(network.incidentEdges(node), adjacentNodeFn(network, node));
      return network.allowsParallelEdges()
          ? UndirectedMultiNetworkConnections.ofImmutable(incidentEdgeMap)
          : UndirectedNetworkConnections.ofImmutable(incidentEdgeMap);
    }
  }

  private static <N, E> Function<E, N> sourceNodeFn(Network<N, E> network) {
    return (E edge) -> network.incidentNodes(edge).source();
  }

  private static <N, E> Function<E, N> targetNodeFn(Network<N, E> network) {
    return (E edge) -> network.incidentNodes(edge).target();
  }

  private static <N, E> Function<E, N> adjacentNodeFn(Network<N, E> network, N node) {
    return (E edge) -> network.incidentNodes(edge).adjacentNode(node);
  }
