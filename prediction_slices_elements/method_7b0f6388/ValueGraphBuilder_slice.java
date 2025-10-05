// Source-based slice around line 189
// Method: <com.google.common.graph.ValueGraphBuilder: MutableValueGraph build()>

    ValueGraphBuilder<N1, V> newBuilder = cast();
    newBuilder.incidentEdgeOrder = checkNotNull(incidentEdgeOrder);
    return newBuilder;
  }

  /**
   * Returns an empty {@link MutableValueGraph} with the properties of this {@link
   * ValueGraphBuilder}.
   */
  public <N1 extends N, V1 extends V> MutableValueGraph<N1, V1> build() {
    return new StandardMutableValueGraph<>(this);
  }

  ValueGraphBuilder<N, V> copy() {
    ValueGraphBuilder<N, V> newBuilder = new ValueGraphBuilder<>(directed);
    newBuilder.allowsSelfLoops = allowsSelfLoops;
    newBuilder.nodeOrder = nodeOrder;
    newBuilder.expectedNodeCount = expectedNodeCount;
    newBuilder.incidentEdgeOrder = incidentEdgeOrder;
    return newBuilder;
