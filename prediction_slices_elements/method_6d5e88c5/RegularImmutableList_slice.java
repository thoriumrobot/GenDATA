// Source-based slice around line 79
// Method: <com.google.common.collect.RegularImmutableList: E get(int)>

  @Override
  int copyIntoArray(@Nullable Object[] dst, int dstOff) {
    arraycopy(array, 0, dst, dstOff, array.length);
    return dstOff + array.length;
  }

  // The fake cast to E is safe because the creation methods only allow E's
  @Override
  @SuppressWarnings("unchecked")
  public E get(int index) {
    return (E) array[index];
  }

  @SuppressWarnings("unchecked")
  @Override
  public UnmodifiableListIterator<E> listIterator(int index) {
    // for performance
    // The fake cast to E is safe because the creation methods only allow E's
    return (UnmodifiableListIterator<E>) Iterators.forArrayWithPosition(array, index);
  }
