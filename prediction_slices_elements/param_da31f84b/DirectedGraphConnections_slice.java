// Source-based slice around line 518
// Method: <com.google.common.graph.DirectedGraphConnections: V addSuccessor(N,V)>


      if (orderedNodeConnections != null) {
        orderedNodeConnections.add(new NodeConnection.Pred<>(node));
      }
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public @Nullable V addSuccessor(N node, V value) {
    Object previousValue = adjacentNodeValues.put(node, value);
    Object previousSuccessor;

    if (previousValue == null) {
      previousSuccessor = null;
    } else if (previousValue instanceof PredAndSucc) {
      adjacentNodeValues.put(node, new PredAndSucc(value));
      previousSuccessor = ((PredAndSucc) previousValue).successorValue;
    } else if (previousValue == PRED) {
      adjacentNodeValues.put(node, new PredAndSucc(value));
