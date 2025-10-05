// Source-based slice around line 309
// Method: <com.google.common.collect.MinMaxPriorityQueue: E elementData(int)>

  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable E poll() {
    return isEmpty() ? null : removeAndGet(0);
  }

  @SuppressWarnings("unchecked") // we must carefully only allow Es to get in
  E elementData(int index) {
    /*
     * requireNonNull is safe as long as we're careful to call this method only with populated
     * indexes.
     */
    return (E) requireNonNull(queue[index]);
  }

  @Override
  public @Nullable E peek() {
    return isEmpty() ? null : elementData(0);
