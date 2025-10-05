// Source-based slice around line 28
// Method: com.google.common.collect.Count.value

import org.jspecify.annotations.Nullable;

/**
 * A mutable value of type {@code int}, for multisets to use in tracking counts of values.
 *
 * @author Louis Wasserman
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
