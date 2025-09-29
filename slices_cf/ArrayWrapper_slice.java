    @Positive
  public @LengthOf("this") int size() {
    return delegate.length;
  }

    @NonNegative
  public void set(@IndexFor("this") int index, T obj) {
    delegate[index] = obj;
  }

  @SuppressWarnings("unchecked") // required for normal Java compilation due to unchecked cast
    @NonNegative
  public T get(@IndexFor("this") int index) {
    return (T) delegate[index];
  }

    @NonNegative
  public static void clearIndex1(ArrayWrapper<? extends Object> a, @IndexFor("#1") int i) {
    a.set(i, null);
  }

    @NonNegative
  public static void clearIndex2(ArrayWrapper<? extends Object> a, int i) {
    if (0 <= i && i < a.size()) {
      a.set(i, null);
    }
  }

    @NonNegative
  public static void clearIndex3(ArrayWrapper<? extends Object> a, @NonNegative int i) {
    if (i < a.size()) {
      a.set(i, null);
    }
  }
