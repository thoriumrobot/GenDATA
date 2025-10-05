// Source-based slice around line 122
// Method: <com.google.common.graph.AbstractValueGraph: boolean equals(Object)>

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
    }
    ValueGraph<?, ?> other = (ValueGraph<?, ?>) obj;

    return isDirected() == other.isDirected()
        && nodes().equals(other.nodes())
