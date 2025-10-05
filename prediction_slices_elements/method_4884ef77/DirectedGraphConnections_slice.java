// Source-based slice around line 489
// Method: <com.google.common.graph.DirectedGraphConnections: void addPredecessor(N,V)>

     *
     * (We promoted a class of warnings into errors because sometimes they indicate real problems.
     * But now we need to "undo" some instance of spurious errors, as discussed in
     * https://github.com/jspecify/checker-framework/issues/8.)
     */
    return removedValue == null ? null : (V) removedValue;
  }

  @Override
  public void addPredecessor(N node, V unused) {
    Object previousValue = adjacentNodeValues.put(node, PRED);
    boolean addedPredecessor;

    if (previousValue == null) {
      addedPredecessor = true;
    } else if (previousValue instanceof PredAndSucc) {
      // Restore previous PredAndSucc object.
      adjacentNodeValues.put(node, previousValue);
      addedPredecessor = false;
    } else if (previousValue != PRED) { // successor
