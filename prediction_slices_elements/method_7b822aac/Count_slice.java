// Source-based slice around line 50
// Method: <com.google.common.collect.Count: int getAndSet(int)>


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
  public int hashCode() {
    return value;
  }

