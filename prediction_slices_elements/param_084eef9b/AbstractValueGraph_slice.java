// Source-based slice around line 112
// Method: <com.google.common.graph.AbstractValueGraph: Optional edgeValue(N,N)>


      @Override
      public int outDegree(N node) {
        return AbstractValueGraph.this.outDegree(node);
      }
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
