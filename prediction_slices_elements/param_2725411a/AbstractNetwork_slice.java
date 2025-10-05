// Source-based slice around line 323
// Method: <com.google.common.graph.AbstractNetwork: Map edgeIncidentNodesMap(Network)>

   * @since 33.1.0
   */
  protected final <T> Set<T> nodePairInvalidatableSet(Set<T> set, N nodeU, N nodeV) {
    return InvalidatableSet.of(
        set,
        () -> nodes().contains(nodeU) && nodes().contains(nodeV),
        () -> String.format(NODE_PAIR_REMOVED_FROM_GRAPH, nodeU, nodeV));
  }

  private static <N, E> Map<E, EndpointPair<N>> edgeIncidentNodesMap(Network<N, E> network) {
    return Maps.asMap(network.edges(), network::incidentNodes);
  }
}
