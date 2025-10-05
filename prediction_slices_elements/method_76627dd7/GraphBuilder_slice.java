// Source-based slice around line 195
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder cast()>

    GraphBuilder<N> newBuilder = new GraphBuilder<>(directed);
    newBuilder.allowsSelfLoops = allowsSelfLoops;
    newBuilder.nodeOrder = nodeOrder;
    newBuilder.expectedNodeCount = expectedNodeCount;
    newBuilder.incidentEdgeOrder = incidentEdgeOrder;
    return newBuilder;
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N> GraphBuilder<N1> cast() {
    return (GraphBuilder<N1>) this;
  }
}
