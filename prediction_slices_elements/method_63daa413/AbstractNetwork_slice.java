// Source-based slice around line 219
// Method: <com.google.common.graph.AbstractNetwork: E edgeConnectingOrNull(EndpointPair)>

        return null;
      case 1:
        return edgesConnecting.iterator().next();
      default:
        throw new IllegalArgumentException(String.format(MULTIPLE_EDGES_CONNECTING, nodeU, nodeV));
    }
  }

  @Override
  public @Nullable E edgeConnectingOrNull(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return edgeConnectingOrNull(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
    checkNotNull(nodeU);
    checkNotNull(nodeV);
    return nodes().contains(nodeU) && successors(nodeU).contains(nodeV);
  }
