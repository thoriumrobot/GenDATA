// Source-based slice around line 42
// Method: <com.google.common.collect.Count: int addAndGet(int)>


  public int get() {
    return value;
  }

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
