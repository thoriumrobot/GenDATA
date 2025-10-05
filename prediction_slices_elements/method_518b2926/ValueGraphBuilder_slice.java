// Source-based slice around line 193
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder copy()>


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
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N, V1 extends V> ValueGraphBuilder<N1, V1> cast() {
