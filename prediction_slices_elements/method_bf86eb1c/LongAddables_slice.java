// Source-based slice around line 27
// Method: <com.google.common.cache.LongAddables: LongAddable create()>

import java.util.concurrent.atomic.LongAdder;

/**
 * Source of {@link LongAddable} objects that deals with GWT and all that.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
final class LongAddables {
  public static LongAddable create() {
    return new JavaUtilConcurrentLongAdder();
  }

  private static final class JavaUtilConcurrentLongAdder extends LongAdder implements LongAddable {}

  private LongAddables() {}
}
