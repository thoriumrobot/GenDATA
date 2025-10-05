// Source-based slice around line 36
// Method: <com.google.common.collect.testing.UnhashableObject: boolean equals(Object)>

@GwtCompatible
public class UnhashableObject implements Comparable<UnhashableObject> {
  private final int value;

  public UnhashableObject(int value) {
    this.value = value;
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof UnhashableObject) {
      UnhashableObject that = (UnhashableObject) object;
      return this.value == that.value;
    }
    return false;
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();
