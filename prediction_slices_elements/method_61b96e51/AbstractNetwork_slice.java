// Source-based slice around line 249
// Method: <com.google.common.graph.AbstractNetwork: boolean isOrderingCompatible(EndpointPair)>

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
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof Network)) {
      return false;
