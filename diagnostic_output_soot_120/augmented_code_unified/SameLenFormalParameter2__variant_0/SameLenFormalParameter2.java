    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenFormalParameter2 {

    @Positive
  void lib(Object @SameLen({"#1", "#2"}) [] valsArg, int @SameLen({"#1", "#2"}) [] modsArg) {}

    @Positive
  void client(Object[] myvals, int[] mymods) {
    // :: error: (argument)
    @Positive
    lib(myvals, mymods);
    @Positive
  }
    @Positive
}
