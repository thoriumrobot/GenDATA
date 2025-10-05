    @Positive
import java.util.ArrayList;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

// @skip-test until we bring list support back

    @Positive
public class ToArrayIndex {

    @Positive
  public String @MinLen(1) [] m(@MinLen(1) ArrayList<String> compiler) {
    @Positive
    return compiler.toArray(new String[0]);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
