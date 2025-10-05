// Source-based slice around line 957
// Method: <com.google.common.collect.MinMaxPriorityQueue: void growIfNeeded()>

    if (initialContents instanceof Collection) {
      int initialSize = ((Collection<?>) initialContents).size();
      result = max(result, initialSize);
    }

    // Now cap it at maxSize + 1
    return capAtMaximumSize(result, maximumSize);
  }

  private void growIfNeeded() {
    if (size > queue.length) {
      int newCapacity = calculateNewCapacity();
      Object[] newQueue = new Object[newCapacity];
      arraycopy(queue, 0, newQueue, 0, queue.length);
      queue = newQueue;
    }
  }

  /** Returns ~2x the old capacity if small; ~1.5x otherwise. */
  private int calculateNewCapacity() {
