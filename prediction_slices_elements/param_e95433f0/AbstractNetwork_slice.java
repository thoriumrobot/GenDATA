// Source-based slice around line 244
// Method: <com.google.common.graph.AbstractNetwork: void validateEndpoints(EndpointPair)>

      return false;
    }
    return hasEdgeConnecting(endpoints.nodeU(), endpoints.nodeV());
  }

  /**
   * Throws an IllegalArgumentException if the ordering of {@code endpoints} is not compatible with
   * the directionality of this graph.
   */
  protected final void validateEndpoints(EndpointPair<?> endpoints) {
    checkNotNull(endpoints);
    checkArgument(isOrderingCompatible(endpoints), ENDPOINTS_MISMATCH);
  }

  protected final boolean isOrderingCompatible(EndpointPair<?> endpoints) {
    return endpoints.isOrdered() == this.isDirected();
  }

  @Override
  public final boolean equals(@Nullable Object obj) {
