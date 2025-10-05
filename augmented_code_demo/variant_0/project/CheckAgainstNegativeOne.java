    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class CheckAgainstNegativeOne {

    @Positive
  public static String replaceString(String target, String oldStr, String newStr) {
    @Positive
    if (oldStr.equals("")) {
    @Positive
      throw new IllegalArgumentException();
    @Positive
    }

    @Positive
    StringBuffer result = new StringBuffer();
    @Positive
    @IndexOrHigh("target") int lastend = 0;
    @Positive
    int pos;
    @Positive
    while ((pos = target.indexOf(oldStr, lastend)) != -1) {
    @Positive
      result.append(target.substring(lastend, pos));
    @Positive
      result.append(newStr);
    @Positive
      lastend = pos + oldStr.length();
    @Positive
    }
    @Positive
    result.append(target.substring(lastend));
    @Positive
    return result.toString();
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
