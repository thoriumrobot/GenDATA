// Source-based slice around line 37
// Method: <com.google.common.collect.CollectPreconditions: int checkNonnegative(int,String)>

  static void checkEntryNotNull(Object key, Object value) {
    if (key == null) {
      throw new NullPointerException("null key in entry: null=" + value);
    } else if (value == null) {
      throw new NullPointerException("null value in entry: " + key + "=null");
    }
  }

  @CanIgnoreReturnValue
  static int checkNonnegative(int value, String name) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " cannot be negative but was: " + value);
    }
    return value;
  }

  @CanIgnoreReturnValue
  static long checkNonnegative(long value, String name) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " cannot be negative but was: " + value);
