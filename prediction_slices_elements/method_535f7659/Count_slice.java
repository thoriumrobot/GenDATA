// Source-based slice around line 46
// Method: <com.google.common.collect.Count: void set(int)>


  public void add(int delta) {
    value += delta;
  }

  public int addAndGet(int delta) {
    return value += delta;
  }

  public void set(int newValue) {
    value = newValue;
  }

  public int getAndSet(int newValue) {
    int result = value;
    value = newValue;
    return result;
  }

  @Override
