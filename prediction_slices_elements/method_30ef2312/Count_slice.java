// Source-based slice around line 34
// Method: <com.google.common.collect.Count: int get()>

 */
@GwtCompatible
final class Count implements Serializable {
  private int value;

  Count(int value) {
    this.value = value;
  }

  public int get() {
    return value;
  }

  public void add(int delta) {
    value += delta;
  }

  public int addAndGet(int delta) {
    return value += delta;
  }
