import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.common.value.qual.MinLen;

public class IndexForTest {

    void test2(@IndexFor("this.array") int i) {
        int x = array[i];
    }
}
