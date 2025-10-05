// Source-based slice around line 667
// Method: <com.google.common.graph.Graphs: long checkPositive(long)>

  }

  @CanIgnoreReturnValue
  static int checkPositive(int value) {
    checkArgument(value > 0, "Not true that %s is positive.", value);
    return value;
  }

  @CanIgnoreReturnValue
  static long checkPositive(long value) {
    checkArgument(value > 0, "Not true that %s is positive.", value);
    return value;
  }

  /**
   * An enum representing the state of a node during DFS. {@code PENDING} means that the node is on
   * the stack of the DFS, while {@code COMPLETE} means that the node and all its successors have
   * been already explored. Any node that has not been explored will not have a state at all.
   */
  private enum NodeVisitState {
