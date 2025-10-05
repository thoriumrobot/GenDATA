// Source-based slice around line 104
// Method: <com.google.common.graph.ImmutableNetwork: NetworkConnections connectionsOf(Network,N)>

    // whatever ordering the network's edges do, so ImmutableSortedMap is unnecessary even if the
    // input edges are sorted.
    ImmutableMap.Builder<E, N> edgeToReferenceNode = ImmutableMap.builder();
    for (E edge : network.edges()) {
      edgeToReferenceNode.put(edge, network.incidentNodes(edge).nodeU());
    }
    return edgeToReferenceNode.buildOrThrow();
  }

  private static <N, E> NetworkConnections<N, E> connectionsOf(Network<N, E> network, N node) {
    if (network.isDirected()) {
      Map<E, N> inEdgeMap = Maps.asMap(network.inEdges(node), sourceNodeFn(network));
      Map<E, N> outEdgeMap = Maps.asMap(network.outEdges(node), targetNodeFn(network));
      int selfLoopCount = network.edgesConnecting(node, node).size();
      return network.allowsParallelEdges()
          ? DirectedMultiNetworkConnections.ofImmutable(inEdgeMap, outEdgeMap, selfLoopCount)
          : DirectedNetworkConnections.ofImmutable(inEdgeMap, outEdgeMap, selfLoopCount);
    } else {
      Map<E, N> incidentEdgeMap =
          Maps.asMap(network.incidentEdges(node), adjacentNodeFn(network, node));
