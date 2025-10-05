// Source-based slice around line 30
// Method: <com.google.common.collect.UsingToStringOrdering: int compare(Object,Object)>

import com.google.common.annotations.J2ktIncompatible;
import java.io.Serializable;

/** An ordering that uses the natural order of the string representation of the values. */
@GwtCompatible
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
