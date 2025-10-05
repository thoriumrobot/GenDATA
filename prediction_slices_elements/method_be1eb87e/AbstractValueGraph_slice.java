// Source-based slice around line 117
// Method: <com.google.common.graph.AbstractValueGraph: Optional edgeValue(EndpointPair)>

    };
  }

  @Override
  public Optional<V> edgeValue(N nodeU, N nodeV) {
    return Optional.ofNullable(edgeValueOrDefault(nodeU, nodeV, null));
  }

  @Override
  public Optional<V> edgeValue(EndpointPair<N> endpoints) {
    return Optional.ofNullable(edgeValueOrDefault(endpoints, null));
  }

  @Override
  public final boolean equals(@Nullable Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof ValueGraph)) {
      return false;
