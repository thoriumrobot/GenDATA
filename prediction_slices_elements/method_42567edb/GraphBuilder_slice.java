// Source-based slice around line 185
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder copy()>

    newBuilder.incidentEdgeOrder = checkNotNull(incidentEdgeOrder);
    return newBuilder;
  }

  /** Returns an empty {@link MutableGraph} with the properties of this {@link GraphBuilder}. */
  public <N1 extends N> MutableGraph<N1> build() {
    return new StandardMutableGraph<>(this);
  }

  GraphBuilder<N> copy() {
    GraphBuilder<N> newBuilder = new GraphBuilder<>(directed);
    newBuilder.allowsSelfLoops = allowsSelfLoops;
    newBuilder.nodeOrder = nodeOrder;
    newBuilder.expectedNodeCount = expectedNodeCount;
    newBuilder.incidentEdgeOrder = incidentEdgeOrder;
    return newBuilder;
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N> GraphBuilder<N1> cast() {
