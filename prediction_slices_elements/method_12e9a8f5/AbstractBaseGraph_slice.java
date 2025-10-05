// Source-based slice around line 179
// Method: <com.google.common.graph.AbstractBaseGraph: void validateEndpoints(EndpointPair)>

    N nodeU = endpoints.nodeU();
    N nodeV = endpoints.nodeV();
    return nodes().contains(nodeU) && successors(nodeU).contains(nodeV);
  }

  /**
   * Throws {@code IllegalArgumentException} if the ordering of {@code endpoints} is not compatible
   * with the directionality of this graph.
   */
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
