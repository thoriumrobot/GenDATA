// Source-based slice around line 35
// Method: <com.google.common.collect.UsingToStringOrdering: Object readResolve()>

final class UsingToStringOrdering extends Ordering<Object> implements Serializable {
  static final UsingToStringOrdering INSTANCE = new UsingToStringOrdering();

  @Override
  public int compare(Object left, Object right) {
    return left.toString().compareTo(right.toString());
  }

  // preserve singleton-ness, so equals() and hashCode() work correctly
  private Object readResolve() {
    return INSTANCE;
  }

  @Override
  public String toString() {
    return "Ordering.usingToString()";
  }

  private UsingToStringOrdering() {}

