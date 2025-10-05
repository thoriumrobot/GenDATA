// Source-based slice around line 455
// Method: <com.google.common.graph.DirectedGraphConnections: V removeSuccessor(Object)>


      if (orderedNodeConnections != null) {
        orderedNodeConnections.remove(new NodeConnection.Pred<>(node));
      }
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public @Nullable V removeSuccessor(Object node) {
    checkNotNull(node);
    Object previousValue = adjacentNodeValues.get(node);
    Object removedValue;

    if (previousValue == null || previousValue == PRED) {
      removedValue = null;
    } else if (previousValue instanceof PredAndSucc) {
      adjacentNodeValues.put((N) node, PRED);
      removedValue = ((PredAndSucc) previousValue).successorValue;
    } else { // successor
