// Source-based slice around line 30
// Method: <com.google.common.base.CommonMatcher: boolean find(int)>

 * shared with the {@code re2j} library. For internal use only. Please refer to the {@code Matcher}
 * javadoc for details.
 */
@GwtCompatible
abstract class CommonMatcher {
  public abstract boolean matches();

  public abstract boolean find();

  public abstract boolean find(int index);

  public abstract String replaceAll(String replacement);

  public abstract int end();

  public abstract int start();
}
