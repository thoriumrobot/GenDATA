// Source-based slice around line 389
// Method: <com.google.common.collect.MinMaxPriorityQueue: E peekLast()>

      throw new NoSuchElementException();
    }
    return removeAndGet(getMaxElementIndex());
  }

  /**
   * Retrieves, but does not remove, the greatest element of this queue, or returns {@code null} if
   * the queue is empty.
   */
  public @Nullable E peekLast() {
    return isEmpty() ? null : elementData(getMaxElementIndex());
  }

  /**
   * Removes the element at position {@code index}.
   *
   * <p>Normally this method leaves the elements at up to {@code index - 1}, inclusive, untouched.
   * Under these circumstances, it returns {@code null}.
   *
   * <p>Occasionally, in order to maintain the heap invariant, it must swap a later element of the
