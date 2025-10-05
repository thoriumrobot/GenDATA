// Source-based slice around line 206
// Method: <com.google.common.graph.AbstractNetwork: E edgeConnectingOrNull(N,N)>

  }

  @Override
  public Optional<E> edgeConnecting(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return edgeConnecting(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public @Nullable E edgeConnectingOrNull(N nodeU, N nodeV) {
    Set<E> edgesConnecting = edgesConnecting(nodeU, nodeV);
    switch (edgesConnecting.size()) {
      case 0:
        return null;
      case 1:
        return edgesConnecting.iterator().next();
      default:
        throw new IllegalArgumentException(String.format(MULTIPLE_EDGES_CONNECTING, nodeU, nodeV));
    }
  }
