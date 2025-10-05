// Source-based slice around line 428
// Method: <com.google.common.graph.DirectedGraphConnections: void removePredecessor(N)>

      return null;
    }
    if (value instanceof PredAndSucc) {
      return (V) ((PredAndSucc) value).successorValue;
    }
    return (V) value;
  }

  @Override
  public void removePredecessor(N node) {
    checkNotNull(node);

    Object previousValue = adjacentNodeValues.get(node);
    boolean removedPredecessor;

    if (previousValue == PRED) {
      adjacentNodeValues.remove(node);
      removedPredecessor = true;
    } else if (previousValue instanceof PredAndSucc) {
      adjacentNodeValues.put((N) node, ((PredAndSucc) previousValue).successorValue);
