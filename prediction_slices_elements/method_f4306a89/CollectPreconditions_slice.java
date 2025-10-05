// Source-based slice around line 28
// Method: <com.google.common.collect.CollectPreconditions: void checkEntryNotNull(Object,Object)>

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.GwtCompatible;
import com.google.errorprone.annotations.CanIgnoreReturnValue;

/** Precondition checks useful in collection implementations. */
@GwtCompatible
final class CollectPreconditions {

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
