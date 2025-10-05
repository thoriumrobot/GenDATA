// Source-based slice around line 132
// Method: <com.google.common.graph.StandardValueGraph: boolean hasEdgeConnecting(N,N)>

          @Override
          public Iterator<EndpointPair<N>> iterator() {
            return connections.incidentEdgeIterator(node);
          }
        };
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

