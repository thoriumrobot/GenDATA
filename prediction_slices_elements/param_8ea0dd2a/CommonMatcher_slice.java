// Source-based slice around line 32
// Method: <com.google.common.base.CommonMatcher: String replaceAll(String)>

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
