    @Positive
import java.util.regex.Pattern;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class Split {
    @Positive
  Pattern p = Pattern.compile(".*");

    @Positive
  void test() {
    @Positive
    String @MinLen(1) [] s = p.split("sdf");
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
