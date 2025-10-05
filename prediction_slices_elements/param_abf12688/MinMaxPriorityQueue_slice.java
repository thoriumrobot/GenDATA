// Source-based slice around line 975
// Method: <com.google.common.collect.MinMaxPriorityQueue: int capAtMaximumSize(int,int)>

  /** Returns ~2x the old capacity if small; ~1.5x otherwise. */
  private int calculateNewCapacity() {
    int oldCapacity = queue.length;
    int newCapacity =
        (oldCapacity < 64) ? (oldCapacity + 1) * 2 : Math.multiplyExact(oldCapacity / 2, 3);
    return capAtMaximumSize(newCapacity, maximumSize);
  }

  /** There's no reason for the queueSize to ever be more than maxSize + 1 */
  private static int capAtMaximumSize(int queueSize, int maximumSize) {
    return min(queueSize - 1, maximumSize) + 1; // don't overflow
  }
}
