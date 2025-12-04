    @Positive
import java.util.ArrayList;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

// @skip-test until we bring list support back

    @Positive
public class ListSupportML {

    @Positive
  void newListMinLen() {
    @Positive
    List<Integer> list = new ArrayList<>();

    // :: error: (assignment)

    @Positive
  }

    @Positive
  void listRemove(@MinLen(10) List<Integer> lst) {
    @Positive
    List<Integer> list = lst;
    @Positive
    list.remove(0);

    // :: error: (assignment)

    @Positive
  }

    @Positive
  void listRemoveAliasing(@MinLen(10) List<Integer> lst) {
    @Positive
    List<Integer> list = lst;

    @Positive
    list2.remove(0);

    // :: error: (assignment)

    @Positive
  }

    @Positive
  void listAdd(@MinLen(10) List<Integer> lst) {
    @Positive
    List<Integer> list = lst;
    @Positive
    list.add(0);

    @Positive
  }

    @Positive
  void listClear(@MinLen(10) List<Integer> lst) {
    @Positive
    List<Integer> list = lst;
    @Positive
    list.clear();

    // :: error: (assignment)

    @Positive
  }

    @Positive
  void listRemoveArrayAlter(@MinLen(10) List<Integer> lst) {
    @Positive
    int[] arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    @Positive
    int @MinLen(10) [] arr1 = arr;
    @Positive
    List<Integer> list = lst;

    @Positive
    list2.remove(0);

    // :: error: (assignment)

    @Positive
    int @MinLen(10) [] arr2 = arr;
    @Positive
  }
    @Positive
}
