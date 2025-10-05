// Source-based slice around line 29
// Method: com.google.common.collect.testing.UnhashableObject.value

import org.jspecify.annotations.Nullable;

/**
 * An unhashable object to be used in testing as values in our collections.
 *
 * @author Regina O'Dell
 */
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
