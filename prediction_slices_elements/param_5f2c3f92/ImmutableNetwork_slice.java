// Source-based slice around line 93
// Method: <com.google.common.graph.ImmutableNetwork: Map getEdgeToReferenceNode(Network)>

    // whatever ordering the network's nodes do, so ImmutableSortedMap is unnecessary even if the
    // input nodes are sorted.
    ImmutableMap.Builder<N, NetworkConnections<N, E>> nodeConnections = ImmutableMap.builder();
    for (N node : network.nodes()) {
      nodeConnections.put(node, connectionsOf(network, node));
    }
    return nodeConnections.buildOrThrow();
  }

  private static <N, E> Map<E, N> getEdgeToReferenceNode(Network<N, E> network) {
    // ImmutableMap.Builder maintains the order of the elements as inserted, so the map will have
    // whatever ordering the network's edges do, so ImmutableSortedMap is unnecessary even if the
    // input edges are sorted.
    ImmutableMap.Builder<E, N> edgeToReferenceNode = ImmutableMap.builder();
    for (E edge : network.edges()) {
      edgeToReferenceNode.put(edge, network.incidentNodes(edge).nodeU());
    }
    return edgeToReferenceNode.buildOrThrow();
  }

