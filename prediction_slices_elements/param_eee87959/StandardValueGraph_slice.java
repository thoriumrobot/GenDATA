// Source-based slice around line 137
// Method: <com.google.common.graph.StandardValueGraph: boolean hasEdgeConnecting(EndpointPair)>

    return nodeInvalidatableSet(incident, node);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
    return hasEdgeConnectingInternal(checkNotNull(nodeU), checkNotNull(nodeV));
  }

  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
    checkNotNull(endpoints);
    return isOrderingCompatible(endpoints)
        && hasEdgeConnectingInternal(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public @Nullable V edgeValueOrDefault(N nodeU, N nodeV, @Nullable V defaultValue) {
    return edgeValueOrDefaultInternal(checkNotNull(nodeU), checkNotNull(nodeV), defaultValue);
  }

