// Source-based slice around line 188
// Method: <com.google.common.graph.AbstractBaseGraph: boolean isOrderingCompatible(EndpointPair)>

  protected final void validateEndpoints(EndpointPair<?> endpoints) {
    checkNotNull(endpoints);
    checkArgument(isOrderingCompatible(endpoints), ENDPOINTS_MISMATCH);
  }

  /**
   * Returns {@code true} iff {@code endpoints}' ordering is compatible with the directionality of
   * this graph.
   */
  protected final boolean isOrderingCompatible(EndpointPair<?> endpoints) {
    return endpoints.isOrdered() == this.isDirected();
  }

  protected final <T> Set<T> nodeInvalidatableSet(Set<T> set, N node) {
    return InvalidatableSet.of(
        set, () -> nodes().contains(node), () -> String.format(NODE_REMOVED_FROM_GRAPH, node));
  }

  protected final <T> Set<T> nodePairInvalidatableSet(Set<T> set, N nodeU, N nodeV) {
    return InvalidatableSet.of(
