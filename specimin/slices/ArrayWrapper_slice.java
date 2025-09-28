  public @LengthOf("this") int size() {
    return delegate.length;
  }

  public void set(@IndexFor("this") int index, T obj) {
    delegate[index] = obj;
  }

  @SuppressWarnings("unchecked") // required for normal Java compilation due to unchecked cast
  public T get(@IndexFor("this") int index) {
    return (T) delegate[index];
  }

  public static void clearIndex1(ArrayWrapper<? extends Object> a, @IndexFor("#1") int i) {
    a.set(i, null);
  }

  public static void clearIndex2(ArrayWrapper<? extends Object> a, int i) {
    if (0 <= i && i < a.size()) {
      a.set(i, null);
    }
