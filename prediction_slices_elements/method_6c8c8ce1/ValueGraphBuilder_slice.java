// Source-based slice around line 203
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder cast()>

    ValueGraphBuilder<N, V> newBuilder = new ValueGraphBuilder<>(directed);
    newBuilder.allowsSelfLoops = allowsSelfLoops;
    newBuilder.nodeOrder = nodeOrder;
    newBuilder.expectedNodeCount = expectedNodeCount;
    newBuilder.incidentEdgeOrder = incidentEdgeOrder;
    return newBuilder;
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N, V1 extends V> ValueGraphBuilder<N1, V1> cast() {
    return (ValueGraphBuilder<N1, V1>) this;
  }
}
